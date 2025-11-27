import axios, { type AxiosInstance, type AxiosRequestConfig } from 'axios'

// Define API response types
export interface GenerateRequest {
  owner: string
  repo: string
  platform?: string
  need_update?: boolean
  branch_mode?: string
  mode?: string
  max_workers?: number
  log?: boolean
}

export interface BaseResponse<T = unknown> {
  code: number
  message: string
  data: T
}

interface WikiInfo {
  owner: string
  repo: string
  wiki_path: string
  wiki_url: string
  total_files: number
  generated_at: string
}

export interface FileInfo {
  name: string
  path: string
  url: string
  size: number
}

export interface GenerateResponseData {
  owner: string
  repo: string
  wiki_path: string
  wiki_url: string
  files: FileInfo[]
  total_files: number
}

interface ListResponse {
  wikis: WikiInfo[]
  total_wikis: number
}

// Create axios instance
const instance: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1', // Default to new API base
  timeout: 10000, // Request timeout
})

// Request interceptor
instance.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
instance.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Generic request function
export const request = async <T = unknown>(
  config: AxiosRequestConfig
): Promise<BaseResponse<T>> => {
  const response = await instance(config)
  return response.data
}

// API call functions
export const generateDoc = async (
  data: GenerateRequest
): Promise<BaseResponse<GenerateResponseData>> => {
  return request<GenerateResponseData>({ method: 'POST', url: '/agents/generate', data })
}

export const listDocs = async (): Promise<BaseResponse<ListResponse>> => {
  return request<ListResponse>({ method: 'GET', url: '/agents/list' })
}

export const generateDocStream = async (
  data: GenerateRequest,
  onMessage: (event: BaseResponse<unknown>) => void | Promise<void>,
  options: { signal?: AbortSignal } = {}
): Promise<BaseResponse<unknown> | undefined> => {
  const baseURL = instance.defaults.baseURL || ''
  const response = await fetch(`${baseURL}/agents/generate-stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(data),
    signal: options.signal,
  })

  if (!response.ok) {
    throw new Error(`Stream request failed with status ${response.status}`)
  }

  const contentType = response.headers.get('content-type') || ''
  let lastEvent: BaseResponse<unknown> | undefined

  if (contentType.includes('application/json')) {
    lastEvent = (await response.json()) as BaseResponse<unknown>
    await onMessage(lastEvent)
    return lastEvent
  }

  const reader = response.body?.getReader()
  if (!reader) {
    return undefined
  }

  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let boundary = buffer.indexOf('\n\n')
      while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary).trim()
        buffer = buffer.slice(boundary + 2)

        if (rawEvent.startsWith('data:')) {
          const payload = rawEvent.replace(/^data:\s*/, '')
          if (payload) {
            try {
              const parsed = JSON.parse(payload) as BaseResponse<unknown>
              lastEvent = parsed
              await onMessage(parsed)
            } catch (err) {
              console.error('Failed to parse stream payload:', err)
            }
          }
        }

        boundary = buffer.indexOf('\n\n')
      }
    }

    buffer += decoder.decode()

    const remaining = buffer.trim()
    if (remaining.startsWith('data:')) {
      const payload = remaining.replace(/^data:\s*/, '')
      if (payload) {
        try {
          const parsed = JSON.parse(payload) as BaseResponse<unknown>
          lastEvent = parsed
          await onMessage(parsed)
        } catch (err) {
          console.error('Failed to parse trailing stream payload:', err)
        }
      }
    }
  } catch (err) {
    if (options.signal?.aborted) {
      const abortError = err as { name?: string }
      if (err instanceof DOMException && err.name === 'AbortError') {
        return lastEvent
      }
      if (abortError?.name === 'AbortError') {
        return lastEvent
      }
    }
    throw err
  } finally {
    reader.releaseLock()
  }

  return lastEvent
}

// Resolve backend static file URL (e.g. "/wikis/..") to full origin.
export const resolveBackendStaticUrl = (path: string) => {
  const base = instance.defaults.baseURL || ''
  let origin = base
  try {
    const u = new URL(base)
    origin = u.origin
  } catch (err) {
    // fallback: strip possible /api/v1 suffix
    origin = base.replace(/\/api\/v1\/?$/, '')
  }

  if (!path) return origin
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  if (path.startsWith('/')) return `${origin}${path}`
  return `${origin}/${path}`
}
