import './App.css'
import { useEffect, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'

type Classroom = { id: number; name: string; created_at: string }
type Student = { id: number; student_no: string; name: string; created_at: string }
type SessionRecord = {
  id: number
  classroom_id: number
  title: string
  started_at: string
  ended_at: string | null
}

type WsEvent =
  | { type: 'attention'; payload: AttentionPayload; ts: string }
  | { type: 'attendance'; payload: AttendancePayload; ts: string }
  | { type: 'session'; payload: SessionPayload; ts: string }

type AttentionPayload = {
  id: number
  session_id: number
  student_id: number
  score_attention: number
  score_expression: number
  score_headpose: number
  score_behavior: number
  ear?: number | null
  mar?: number | null
  yaw?: number | null
  pitch?: number | null
  roll?: number | null
  ts: string
}

type AttendancePayload = {
  id?: number
  session_id: number
  student_id: number
  status: 'present' | 'late' | 'absent'
  ts: string
}

type SessionPayload = { event: 'started' | 'ended'; session: SessionRecord }
type DemoStartOut = {
  ok: boolean
  classroom_id: number
  session_id: number
  student_ids?: number[]
  student_count?: number
}
type SessionEndOut = SessionRecord & {
  attendance_summary?: {
    roster_count: number
    present: number
    absent: number
    late: number
    marked_absent: number
  }
}
type CameraStartOut = {
  ok: boolean
  session_id: number
  classroom_id: number
  student_ids: number[]
  student_count: number
  message: string
}
type CameraStatusOut = {
  running: boolean
  pid: number | null
  session_id: number | null
  classroom_id: number | null
  student_ids: number[]
}

function attendanceLabel(status?: AttendancePayload['status']): string {
  if (status === 'present') return '已到'
  if (status === 'late') return '迟到'
  if (status === 'absent') return '缺席'
  return '未签'
}

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

/** Direct ws://host:8001 is more reliable than Vite WS proxy on customer Windows. */
function resolveWsBase(): string {
  const envWs = import.meta.env.VITE_WS_BASE as string | undefined
  if (envWs?.trim()) return envWs.trim().replace(/\/$/, '')
  const envApi = import.meta.env.VITE_API_BASE as string | undefined
  if (envApi?.trim()) return envApi.trim().replace(/^http/i, 'ws').replace(/\/$/, '')
  const host = window.location.hostname
  if (host === '127.0.0.1' || host === 'localhost') return 'ws://127.0.0.1:8001'
  if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) return `ws://${host}:8001`
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}`
}

const WS_BASE = resolveWsBase()
const CAMERA_API_BASE =
  import.meta.env.VITE_CAMERA_API_BASE ?? (API_BASE || 'http://127.0.0.1:8001')

function wsStatusLabel(s: 'disconnected' | 'connecting' | 'connected'): string {
  if (s === 'connected') return '已连接'
  if (s === 'connecting') return '连接中'
  return '已断开'
}

function resolveApiBaseForProbe(): string {
  const env = import.meta.env.VITE_API_BASE as string | undefined
  if (env?.trim()) return env.trim().replace(/\/$/, '')
  const host = window.location.hostname
  if (host === '127.0.0.1' || host === 'localhost') return 'http://127.0.0.1:8001'
  if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) return `http://${host}:8001`
  return ''
}

async function waitBackendReady(maxMs = 20000): Promise<boolean> {
  const base = resolveApiBaseForProbe()
  const healthUrl = base ? `${base}/api/health` : '/api/health'
  const deadline = Date.now() + maxMs
  while (Date.now() < deadline) {
    try {
      const r = await fetch(healthUrl)
      if (r.ok) return true
    } catch {
      /* retry */
    }
    await new Promise((r) => setTimeout(r, 600))
  }
  return false
}

async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return (await r.json()) as T
}
async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`
    try {
      const err = (await r.json()) as { detail?: string }
      if (err.detail) detail = err.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return (await r.json()) as T
}

function App() {
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>(
    'disconnected',
  )
  const [error, setError] = useState<string | null>(null)

  const [classrooms, setClassrooms] = useState<Classroom[]>([])
  const [students, setStudents] = useState<Student[]>([])

  const [classroomId, setClassroomId] = useState<number>(1)
  const [sessionId, setSessionId] = useState<number | null>(null)

  const [attendance, setAttendance] = useState<Record<number, AttendancePayload>>({})
  const [latestAttention, setLatestAttention] = useState<Record<number, AttentionPayload>>({})
  const [selectedStudentId, setSelectedStudentId] = useState<number | null>(null)
  const [history, setHistory] = useState<AttentionPayload[]>([])
  const [cameraStudentCount, setCameraStudentCount] = useState(0)
  const [cameraRunning, setCameraRunning] = useState(false)
  const [showAddStudent, setShowAddStudent] = useState(false)
  const [newStudentNo, setNewStudentNo] = useState('')
  const [newStudentName, setNewStudentName] = useState('')
  const [addingStudent, setAddingStudent] = useState(false)
  const [sessionRosterIds, setSessionRosterIds] = useState<number[]>([])
  const [endSummary, setEndSummary] = useState<string | null>(null)
  const [wsReconnectKey, setWsReconnectKey] = useState(0)

  const wsRef = useRef<WebSocket | null>(null)
  const selectedStudentIdRef = useRef<number | null>(null)
  const sessionIdRef = useRef<number | null>(null)

  async function syncCameraStatus() {
    try {
      const st = await apiGet<CameraStatusOut>('/api/camera/status')
      setCameraRunning(st.running)
      setCameraStudentCount(st.student_ids?.length ?? 0)
      if (st.running && st.classroom_id && st.session_id) {
        setClassroomId(st.classroom_id)
        setSessionId(st.session_id)
      }
      if (st.student_ids?.length) {
        setSessionRosterIds(st.student_ids)
      }
    } catch (e) {
      setError(String(e))
    }
  }

  async function refreshBase() {
    setError(null)
    const [cs, ss] = await Promise.all([
      apiGet<Classroom[]>('/api/classrooms'),
      apiGet<Student[]>('/api/students'),
    ])
    setClassrooms(cs)
    setStudents(ss)
    if (cs.length > 0) {
      const has = cs.some((c) => c.id === classroomId)
      if (!has) setClassroomId(cs[0].id)
    }
    if (!selectedStudentId && ss.length > 0) {
      setSelectedStudentId(ss[0].id)
    }
  }

  useEffect(() => {
    refreshBase().catch((e) => setError(String(e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classroomId])

  useEffect(() => {
    // keep UI aligned with the running camera process
    const t = window.setInterval(() => {
      syncCameraStatus().catch(() => {})
    }, 2000)
    return () => window.clearInterval(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!selectedStudentId || !sessionId) {
      setHistory([])
      return
    }
    const pullHistory = async () => {
      try {
        const rows = await apiGet<AttentionPayload[]>(
          `/api/attention/history?session_id=${sessionId}&student_id=${selectedStudentId}&limit=600`,
        )
        setHistory(rows)
      } catch (e) {
        setError(String(e))
      }
    }

    // Initial load when session/student changes.
    void pullHistory()
    // Polling fallback: keeps curve updating even if WS hiccups.
    const timer = window.setInterval(() => {
      void pullHistory()
    }, 2000)
    return () => window.clearInterval(timer)
  }, [selectedStudentId, sessionId])

  useEffect(() => {
    if (!sessionId) return
    const pullAttendance = async () => {
      try {
        const rows = await apiGet<AttendancePayload[]>(`/api/attendance/latest?session_id=${sessionId}`)
        if (!rows || rows.length === 0) return
        setAttendance((prev) => {
          const next = { ...prev }
          for (const r of rows) next[r.student_id] = r
          return next
        })
      } catch {
        // Ignore transient errors, WS still updates in realtime.
      }
    }
    const pullLatest = async () => {
      try {
        const rows = await apiGet<AttentionPayload[]>(`/api/attention/latest?session_id=${sessionId}`)
        if (!rows || rows.length === 0) return
        setLatestAttention((prev) => {
          const next = { ...prev }
          for (const r of rows) next[r.student_id] = r
          return next
        })
      } catch {
        // Keep silent; WS path still works when available.
      }
    }
    void pullAttendance()
    void pullLatest()
    const timer = window.setInterval(() => {
      void pullAttendance()
      void pullLatest()
    }, 2000)
    return () => window.clearInterval(timer)
  }, [sessionId])

  useEffect(() => {
    selectedStudentIdRef.current = selectedStudentId
  }, [selectedStudentId])

  useEffect(() => {
    sessionIdRef.current = sessionId
  }, [sessionId])

  useEffect(() => {
    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let attempt = 0

    const onMessage = (ev: MessageEvent) => {
      const msg = JSON.parse(ev.data) as WsEvent
      if (msg.type === 'attendance') {
        const p = msg.payload
        setAttendance((prev) => ({ ...prev, [p.student_id]: p }))
      } else if (msg.type === 'attention') {
        const p = msg.payload
        setLatestAttention((prev) => ({ ...prev, [p.student_id]: p }))
        if (selectedStudentIdRef.current === p.student_id) {
          setHistory((prev) => [...prev.slice(-599), p])
        }
      } else if (msg.type === 'session') {
        refreshBase().catch((e) => setError(String(e)))
        const s = msg.payload.session
        if (msg.payload.event === 'started') setSessionId(s.id)
        if (msg.payload.event === 'ended' && sessionIdRef.current === s.id) setSessionId(null)
      }
    }

    const connect = () => {
      if (cancelled) return
      setStatus('connecting')
      wsRef.current?.close()
      const url = `${WS_BASE}/ws/classrooms/${classroomId}`
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        attempt = 0
        setStatus('connected')
        ws.send('hello')
      }
      ws.onerror = () => {
        if (!cancelled) setStatus('disconnected')
      }
      ws.onclose = () => {
        if (cancelled) return
        setStatus('disconnected')
        attempt += 1
        const delay = Math.min(5000, 800 + attempt * 600)
        retryTimer = setTimeout(connect, delay)
      }
      ws.onmessage = onMessage
    }

    void (async () => {
      const ok = await waitBackendReady()
      if (cancelled) return
      if (!ok) {
        setStatus('disconnected')
        setError('后端未就绪：请确认 TeachingAssist-Backend 窗口在运行，然后点「重连WS」')
        return
      }
      connect()
    })()

    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
      wsRef.current?.close()
      wsRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [classroomId, wsReconnectKey])

  const studentRows = useMemo(() => {
    return students.map((s) => {
      const att = latestAttention[s.id]
      const atd = attendance[s.id]
      return { s, att, atd }
    })
  }, [students, latestAttention, attendance])

  const rosterIdsForStats = useMemo(() => {
    if (sessionRosterIds.length > 0) return sessionRosterIds
    return students.map((s) => s.id)
  }, [sessionRosterIds, students])

  const overview = useMemo(() => {
    const total = rosterIdsForStats.length
    const signed = rosterIdsForStats.filter((id) => attendance[id]?.status === 'present').length
    const absent = rosterIdsForStats.filter((id) => attendance[id]?.status === 'absent').length
    const rosterAttention = rosterIdsForStats
      .map((id) => latestAttention[id]?.score_attention)
      .filter((x): x is number => x != null)
    const avgAttention =
      rosterAttention.length > 0
        ? rosterAttention.reduce((a, b) => a + b, 0) / rosterAttention.length
        : null
    return { total, signed, absent, avgAttention }
  }, [rosterIdsForStats, attendance, latestAttention])

  const chartOption = useMemo(() => {
    const xs = history.map((h) => new Date(h.ts).toLocaleTimeString())
    const ys = history.map((h) => Math.round(h.score_attention * 10) / 10)
    const smallSeries = ys.length <= 2
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 40, right: 20, top: 30, bottom: 40 },
      xAxis: { type: 'category', data: xs },
      yAxis: { type: 'value', min: 0, max: 100 },
      series: [
        {
          type: 'line',
          data: ys,
          smooth: true,
          showSymbol: smallSeries,
          symbolSize: smallSeries ? 8 : 4,
          connectNulls: true,
        },
      ],
    }
  }, [history])

  async function seedDemo() {
    setError(null)
    setEndSummary(null)
    const out = await apiPost<DemoStartOut>('/api/demo/start', {})
    setAttendance({})
    setLatestAttention({})
    setHistory([])
    setSelectedStudentId(null)
    setClassroomId(out.classroom_id)
    setSessionId(out.session_id)
    setSessionRosterIds(out.student_ids ?? [])
    setCameraStudentCount(out.student_count ?? out.student_ids?.length ?? 0)
  }

  async function startSession() {
    if (!classroomId) return
    setError(null)
    setEndSummary(null)
    const maxStudents = Math.max(1, Math.min(students.length || 4, 20))
    const s = await apiPost<CameraStartOut>('/api/camera/start', {
      classroom_id: classroomId,
      max_students: maxStudents,
      api_base: CAMERA_API_BASE,
    })
    setClassroomId(s.classroom_id)
    setSessionId(s.session_id)
    setSessionRosterIds(s.student_ids ?? [])
    setSelectedStudentId((prev) => prev ?? s.student_ids[0] ?? null)
    setCameraStudentCount(s.student_count ?? s.student_ids.length)
    setCameraRunning(true)
    await refreshBase()
  }

  async function endSession() {
    if (!sessionId) return
    setError(null)
    const endingSessionId = sessionId
    try {
      await apiPost<{ ok: boolean }>('/api/camera/stop', {})
      try {
        await apiPost<{ ok: boolean }>('/api/demo/stop', {})
      } catch {
        /* demo may not be running */
      }
      const ended = await apiPost<SessionEndOut>(`/api/sessions/${endingSessionId}/end`, {})
      const rows = await apiGet<AttendancePayload[]>(
        `/api/attendance/latest?session_id=${endingSessionId}`,
      )
      const next: Record<number, AttendancePayload> = {}
      for (const r of rows) next[r.student_id] = r
      setAttendance(next)
      const sum = ended.attendance_summary
      if (sum) {
        setEndSummary(
          `下课统计：本节课 ${sum.roster_count} 人，已到 ${sum.present} 人，缺席 ${sum.absent} 人（未检测到人脸的自动记缺席）`,
        )
      }
    } catch (e) {
      setError(String(e))
      return
    }
    setSessionId(null)
    setSessionRosterIds([])
    setCameraRunning(false)
    await refreshBase()
  }

  async function addStudent() {
    const student_no = newStudentNo.trim()
    const name = newStudentName.trim()
    if (!student_no || !name) {
      setError('请填写学号和姓名')
      return
    }
    setError(null)
    setAddingStudent(true)
    try {
      await apiPost<Student>('/api/students', { student_no, name })
      setNewStudentNo('')
      setNewStudentName('')
      setShowAddStudent(false)
      await refreshBase()
      if (cameraRunning) {
        setError('学生已添加。上课中需点击「重启摄像头」后，新学生才会进入多人识别。')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setAddingStudent(false)
    }
  }

  async function restartCamera() {
    if (!sessionId) {
      setError('请先开始上课')
      return
    }
    setError(null)
    try {
      const maxStudents = Math.max(1, Math.min(students.length || 4, 20))
      const s = await apiPost<CameraStartOut>('/api/camera/restart', {
        classroom_id: classroomId,
        max_students: maxStudents,
        api_base: CAMERA_API_BASE,
      })
      setSessionRosterIds(s.student_ids ?? [])
      setCameraStudentCount(s.student_count ?? s.student_ids.length)
      setCameraRunning(true)
      await refreshBase()
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <div className="title">教学管理辅助系统（本地演示）</div>
          <div className="sub">REST + WebSocket 实时看板（后续可平移到小程序）</div>
        </div>
        <div className="pillrow">
          <span className={`pill ${status}`}>WS: {wsStatusLabel(status)}</span>
          <span className="pill">API: {API_BASE || '同页代理'}</span>
          {status === 'disconnected' ? (
            <span className="pill disconnected">请确认 Backend 窗口在运行 (8001)</span>
          ) : null}
        </div>
      </header>

      <section className="grid">
        <div className="card">
          <div className="cardTitle">课堂控制</div>
          <div className="row">
            <label>教室</label>
            <select
              value={classroomId}
              onChange={(e) => setClassroomId(Number(e.target.value))}
            >
              {classrooms.length === 0 ? <option value={1}>（暂无）</option> : null}
              {classrooms.map((c) => (
                <option key={c.id} value={c.id}>
                  #{c.id} {c.name}
                </option>
              ))}
            </select>
            <button
              onClick={() => setWsReconnectKey((k) => k + 1)}
              disabled={status === 'connecting'}
              title="触发 WebSocket 重新连接"
            >
              重连WS
            </button>
          </div>

          <div className="row">
            <button onClick={() => seedDemo()}>一键生成演示数据</button>
            <button onClick={() => startSession()}>开始上课（多人识别）</button>
            <button onClick={() => restartCamera()} disabled={!cameraRunning || !sessionId}>
              重启摄像头
            </button>
            <button onClick={() => syncCameraStatus()}>同步摄像头状态</button>
            <button onClick={() => endSession()} disabled={!sessionId}>
              下课/结束
            </button>
          </div>

          <div className="muted">
            当前 Session: {sessionId ?? '（未开始）'}；摄像头多人识别名额: {cameraStudentCount}；
            点击学生仅影响曲线查看（摄像头会并行识别多张人脸并上报）。
          </div>
          <div className="row">
            <span className="tag status-present">已到 {overview.signed}</span>
            <span className="tag status-absent">缺席 {overview.absent}</span>
            <span className="tag">本节课 {overview.total} 人</span>
            <span className="tag">
              平均专注度 {overview.avgAttention === null ? '-' : Math.round(overview.avgAttention)}
            </span>
          </div>
          {sessionId ? (
            <div className="muted">
              签到说明：摄像头识别到人脸记为「已到」；点击「下课/结束」后，本节课名单中未识别到的学生自动记「缺席」。
            </div>
          ) : null}
          {endSummary ? <div className="info">{endSummary}</div> : null}

          {error ? <div className="error">错误：{error}</div> : null}
        </div>

        <div className="card">
          <div className="cardTitle row between">
            <span>学生实时列表</span>
            <button type="button" className="btnSmall" onClick={() => setShowAddStudent((v) => !v)}>
              {showAddStudent ? '取消' : '+ 添加学生'}
            </button>
          </div>
          {showAddStudent ? (
            <div className="addStudentForm">
              <div className="row">
                <label>学号</label>
                <input
                  value={newStudentNo}
                  onChange={(e) => setNewStudentNo(e.target.value)}
                  placeholder="如 2026005"
                  maxLength={32}
                />
              </div>
              <div className="row">
                <label>姓名</label>
                <input
                  value={newStudentName}
                  onChange={(e) => setNewStudentName(e.target.value)}
                  placeholder="如 小明"
                  maxLength={64}
                />
              </div>
              <div className="row">
                <button type="button" onClick={() => addStudent()} disabled={addingStudent}>
                  {addingStudent ? '保存中…' : '保存学生'}
                </button>
              </div>
              <div className="muted">
                上课中添加学生后，需点击「重启摄像头」，新学生才会出现在本机摄像头窗口并参与识别（按从左到右座位顺序对应学号列表）。
              </div>
            </div>
          ) : null}
          <div className="muted">
            点击某个学生 = 查看他的曲线。当前摄像头为多人识别模式，同一画面内多位学生可同时上报实时数据。
          </div>
          <div className="table">
            <div className="thead">
              <div>学生</div>
              <div>签到</div>
              <div>专注度</div>
              <div>子分</div>
              <div>证据</div>
            </div>
            {studentRows.map(({ s, att, atd }) => (
              <button
                key={s.id}
                className={`trow ${selectedStudentId === s.id ? 'active' : ''}`}
                onClick={() => setSelectedStudentId(s.id)}
              >
                <div>
                  <div className="strong">
                    {s.name} <span className="muted">({s.student_no})</span>
                  </div>
                  <div className="muted">id: {s.id}</div>
                </div>
                <div className={`tag status-${atd?.status ?? 'none'}`}>{attendanceLabel(atd?.status)}</div>
                <div className="score">{att ? Math.round(att.score_attention) : '-'}</div>
                <div className="muted">
                  {att
                    ? `表情 ${Math.round(att.score_expression)} / 头姿 ${Math.round(att.score_headpose)} / 行为 ${Math.round(att.score_behavior)}`
                    : '-'}
                </div>
                <div className="muted">
                  {att
                    ? `EAR ${att.ear ?? '-'} MAR ${att.mar ?? '-'} yaw ${att.yaw ?? '-'}`
                    : '-'}
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="card span2">
          <div className="cardTitle">专注度曲线（选中学生）</div>
          <div className="muted">
            学生ID: {selectedStudentId ?? '未选择'}；点上方“学生实时列表”中的行即可查看实时曲线。
          </div>
          <div className="muted">当前曲线点数: {history.length}</div>
          <div className="chart">
            <ReactECharts option={chartOption} style={{ height: 320, width: '100%' }} />
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="muted">
          后端：FastAPI + SQLite；前端：Vite + React + ECharts；实时：WebSocket（本地演示不依赖外网）。
        </div>
      </footer>
    </div>
  )
}

export default App
