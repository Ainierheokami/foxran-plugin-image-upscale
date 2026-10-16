import type { Component } from 'vue'

/** What the Foxran WebUI exposes to extension bundles on window.Foxran. */
type FoxranHost = {
  registerPlugin(plugin: unknown): void
  api: {
    get<T = any>(url: string): Promise<{ data: T }>
    post<T = any>(url: string, body?: unknown): Promise<{ data: T }>
  }
  ui: {
    AdapterFormSection: Component
    toast?: { success(message: string): void; error(message: string): void }
  }
}

export const host = (window as any).Foxran as FoxranHost
export const { AdapterFormSection } = host.ui
