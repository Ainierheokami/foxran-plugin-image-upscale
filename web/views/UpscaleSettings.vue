<template>
  <div class="ihu-page">
    <header class="ihu-head">
      <div class="ihu-head-text">
        <span class="ihu-file">{{ currentFile || 'image_hd_upscale.yaml' }}</span>
        <h3><ImageUpscale class="ihu-icon" /> 图片放大</h3>
        <p>切块交给图像编辑模型高清重绘，再对齐拼回原图。保存后下一次放大立即生效。</p>
      </div>
      <div class="ihu-head-actions">
        <slot name="header-actions" />
        <button class="ihu-btn" :disabled="busy" title="重新读取" @click="load"><RefreshCw :class="['ihu-btn-icon', busy && 'ihu-spin']" /> 刷新</button>
        <button class="ihu-btn ihu-btn-primary" :disabled="busy || !dirty" @click="save"><Save class="ihu-btn-icon" /> 保存</button>
      </div>
    </header>

    <div v-if="error" class="ihu-alert ihu-alert-error">{{ error }}</div>
    <div v-else-if="message" class="ihu-alert">{{ message }}</div>

    <div v-if="cfg" class="ihu-sections">
      <AdapterFormSection index="01" title="模型" description="重绘每一块用的图像编辑模型，以及放大前校对文字用的视觉模型。" :icon="Bot">
        <label>
          <span>重绘模型</span>
          <select v-model="cfg.model_name" @change="touch">
            <option value="">自动（按 image_edit 标签轮询）</option>
            <option v-for="m in editModels" :key="m.name" :value="m.name">{{ modelLabel(m) }}</option>
            <option v-if="cfg.model_name && !hasModel(editModels, cfg.model_name)" :value="cfg.model_name">{{ cfg.model_name }}（未找到）</option>
          </select>
          <small>{{ desc('model_name') }} 推荐 gpt-image-2。</small>
        </label>
        <label>
          <span>校对模型</span>
          <select v-model="cfg.proofread_model" :disabled="!cfg.proofread" @change="touch">
            <option value="">自动（按 vision 标签选）</option>
            <option v-for="m in visionModels" :key="m.name" :value="m.name">{{ modelLabel(m) }}</option>
            <option v-if="cfg.proofread_model && !hasModel(visionModels, cfg.proofread_model)" :value="cfg.proofread_model">{{ cfg.proofread_model }}（未找到）</option>
          </select>
          <small>{{ desc('proofread_model') }}</small>
        </label>
        <label class="ihu-switch ihu-wide">
          <span><b>放大前校对文字</b><small>先让视觉模型通读整张图，按上下文找出错别字和乱码（如“做蠣启德”应为“坐落启德”），把修正写进每块的提示词。每张图多一次模型调用，失败不影响放大。</small></span>
          <input v-model="cfg.proofread" type="checkbox" @change="touch" />
        </label>
        <label class="ihu-wide">
          <span>附加提示词</span>
          <textarea v-model="cfg.prompt" rows="2" placeholder="例如：保持扁平插画风格" @input="touch" />
          <small>{{ desc('prompt') }}</small>
        </label>
      </AdapterFormSection>

      <AdapterFormSection index="02" title="倍率与切块" description="切块越大、块数越少，模型调用越少；相邻块的冗余用来对齐和找拼缝。" :icon="Grid3x3">
        <label><span>默认倍率</span><input v-model.number="cfg.default_scale" type="number" step="0.25" :min="1" :max="cfg.max_scale" @input="touch" /><small>{{ desc('default_scale') }}</small></label>
        <label><span>最大倍率</span><input v-model.number="cfg.max_scale" type="number" step="0.25" min="1" max="8" @input="touch" /><small>{{ desc('max_scale') }}</small></label>
        <div class="ihu-field ihu-wide">
          <span>切块尺寸</span>
          <div class="ihu-chips">
            <span v-for="(size, i) in cfg.tile_sizes" :key="size + i" class="ihu-chip">
              {{ size }}
              <button type="button" title="移除" :disabled="cfg.tile_sizes.length <= 1" @click="removeSize(i)"><X class="ihu-chip-x" /></button>
            </span>
            <input v-model="newSize" class="ihu-chip-input" placeholder="宽x高，回车添加" @keydown.enter.prevent="addSize" />
          </div>
          <small>{{ desc('tile_sizes') }} GPT Image 2 支持 1024x1024、1536x1024、1024x1536 等。</small>
        </div>
        <label><span>冗余比例</span><input v-model.number="cfg.overlap_ratio" type="number" step="0.01" min="0.02" max="0.45" @input="touch" /><small>{{ desc('overlap_ratio') }}</small></label>
        <label><span>最多块数</span><input v-model.number="cfg.max_tiles" type="number" min="1" max="64" @input="touch" /><small>{{ desc('max_tiles') }}</small></label>
        <label><span>输出像素上限（百万）</span><input :value="megapixels" type="number" step="1" min="1" max="200" @input="setMegapixels(($event.target as HTMLInputElement).valueAsNumber)" /><small>{{ desc('max_output_pixels') }}</small></label>
        <label><span>并发块数</span><input v-model.number="cfg.max_concurrency" type="number" min="1" max="16" @input="touch" /><small>{{ desc('max_concurrency') }}</small></label>
      </AdapterFormSection>

      <AdapterFormSection index="03" title="重试与修正" description="模型重绘后的对齐、偏差检测和校色。默认值来自实机调试，一般不用改。" :icon="Wrench">
        <label><span>单块自动重试次数</span><input v-model.number="cfg.retries" type="number" min="0" max="5" @input="touch" /><small>{{ desc('retries') }} 事后还可以用 /放大重试 单独重画失败的块。</small></label>
        <label><span>偏差占比上限</span><input v-model.number="cfg.max_deviation" type="number" step="0.01" min="0" max="1" @input="touch" /><small>{{ desc('max_deviation') }}</small></label>
        <label><span>结构偏差阈值（0~100）</span><input v-model.number="cfg.layout_threshold" type="number" step="1" min="0" max="100" @input="touch" /><small>{{ desc('layout_threshold') }}</small></label>
        <label><span>校色强度（0~1）</span><input v-model.number="cfg.color_correction" type="number" step="0.05" min="0" max="1" @input="touch" /><small>{{ desc('color_correction') }}</small></label>
        <label><span>拼缝羽化（像素）</span><input v-model.number="cfg.seam_feather" type="number" step="1" min="0" max="64" @input="touch" /><small>{{ desc('seam_feather') }}</small></label>
        <label class="ihu-switch"><span><b>对齐重绘结果</b><small>{{ desc('align') }}</small></span><input v-model="cfg.align" type="checkbox" @change="touch" /></label>
        <label class="ihu-switch"><span><b>光流修正局部挪动</b><small>{{ desc('local_flow') }}</small></span><input v-model="cfg.local_flow" type="checkbox" :disabled="!cfg.align" @change="touch" /></label>
        <label class="ihu-switch"><span><b>失败时用插值顶上</b><small>{{ desc('fallback_on_failure') }}</small></span><input v-model="cfg.fallback_on_failure" type="checkbox" @change="touch" /></label>
      </AdapterFormSection>

      <AdapterFormSection index="04" title="输出与记录" description="结果图的格式，以及供 /放大重试 使用的放大记录。" :icon="HardDrive">
        <label>
          <span>输出格式</span>
          <select v-model="cfg.output_format" @change="touch"><option value="png">PNG</option><option value="jpg">JPG</option><option value="webp">WebP</option></select>
          <small>{{ desc('output_format') }}</small>
        </label>
        <label><span>JPG / WebP 质量</span><input v-model.number="cfg.jpeg_quality" type="number" min="50" max="100" @input="touch" /><small>{{ desc('jpeg_quality') }}</small></label>
        <label><span>PNG 大小上限（MB）</span><input v-model.number="cfg.max_output_mb" type="number" step="1" min="0" @input="touch" /><small>{{ desc('max_output_mb') }}</small></label>
        <label><span>保留放大记录（次）</span><input v-model.number="cfg.keep_jobs" type="number" min="0" @input="touch" /><small>{{ desc('keep_jobs') }} 记录存在 data/image_hd_upscale/jobs/。</small></label>
        <label class="ihu-switch ihu-wide"><span><b>保存调试切块</b><small>{{ desc('save_tiles') }} 效果不对时打开，把对应目录发给开发者排查。</small></span><input v-model="cfg.save_tiles" type="checkbox" @change="touch" /></label>
      </AdapterFormSection>

      <div class="ihu-foot">
        <button class="ihu-btn" :disabled="busy" @click="resetDefaults"><RotateCcw class="ihu-btn-icon" /> 恢复默认值</button>
      </div>
    </div>
    <div v-else-if="busy" class="ihu-loading"><Loader2 class="ihu-spin ihu-loading-icon" /></div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Bot, Grid3x3, HardDrive, ImageUpscale, Loader2, RefreshCw, RotateCcw, Save, Wrench, X } from 'lucide-vue-next'
import { AdapterFormSection, host } from '../host'

defineProps<{ currentFile?: string }>()

type Config = Record<string, any> & { tile_sizes: string[] }
type Model = { name: string; tags: string[]; available: boolean }
type Payload = { config: Config; defaults: Config; descriptions: Record<string, string> }

const API = '/v1/plugins/image-hd-upscale/config'
const cfg = ref<Config | null>(null)
const defaults = ref<Config | null>(null)
const descriptions = ref<Record<string, string>>({})
const models = ref<Model[]>([])
const busy = ref(false)
const dirty = ref(false)
const error = ref('')
const message = ref('')
const newSize = ref('')

const editModels = computed(() => models.value.filter((m) => m.tags.includes('image_edit')))
const visionModels = computed(() => models.value.filter((m) => m.tags.includes('vision')))
const megapixels = computed(() => Math.round((cfg.value?.max_output_pixels || 0) / 1_000_000))

function desc(name: string) { return descriptions.value[name] || '' }
function touch() { dirty.value = true; message.value = '' }
function hasModel(list: Model[], name: string) { return list.some((m) => m.name === name) }
function modelLabel(m: Model) { return m.available ? m.name : `${m.name}（冷却中/不可用）` }
function detail(e: any, fallback: string) { return e?.response?.data?.detail || fallback }

function apply(data: Payload) {
  cfg.value = structuredClone(data.config)
  defaults.value = data.defaults
  descriptions.value = data.descriptions || {}
  dirty.value = false
}

function setMegapixels(value: number) {
  if (!cfg.value || !Number.isFinite(value)) return
  cfg.value.max_output_pixels = Math.round(value * 1_000_000)
  touch()
}

function addSize() {
  if (!cfg.value) return
  const text = newSize.value.trim().toLowerCase().replace(/[×*\s]/g, 'x')
  const match = /^(\d{3,4})x(\d{3,4})$/.exec(text)
  if (!match) { error.value = '切块尺寸应写成 宽x高，如 1536x1024'; return }
  const size = `${Number(match[1])}x${Number(match[2])}`
  if (!cfg.value.tile_sizes.includes(size)) cfg.value.tile_sizes.push(size)
  newSize.value = ''
  error.value = ''
  touch()
}

function removeSize(index: number) {
  if (!cfg.value || cfg.value.tile_sizes.length <= 1) return
  cfg.value.tile_sizes.splice(index, 1)
  touch()
}

function resetDefaults() {
  if (!defaults.value) return
  cfg.value = structuredClone(defaults.value)
  touch()
  message.value = '已恢复默认值，保存后生效。'
}

async function loadModels() {
  try {
    const { data } = await host.api.get<{ models: Model[] }>('/v1/models/catalog')
    models.value = Array.isArray(data.models) ? data.models : []
  } catch {
    models.value = []
  }
}

async function load() {
  busy.value = true
  error.value = ''
  message.value = ''
  try {
    const [{ data }] = await Promise.all([host.api.get<Payload>(API), loadModels()])
    apply(data)
  } catch (e) {
    error.value = detail(e, '无法读取图片放大配置，请确认插件已启用。')
  } finally {
    busy.value = false
  }
}

async function save() {
  if (!cfg.value) return
  busy.value = true
  error.value = ''
  try {
    const { data } = await host.api.post<Payload>(API, { config: cfg.value })
    apply(data)
    message.value = '配置已保存，下一次放大生效。'
    host.ui.toast?.success('图片放大配置已保存')
  } catch (e) {
    error.value = detail(e, '保存失败')
  } finally {
    busy.value = false
  }
}

defineExpose({ load, save })
onMounted(load)
</script>

<style scoped>
.ihu-page { max-width: 64rem; width: 100%; margin: 0 auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; color: hsl(var(--foreground)); }
@media (min-width: 768px) { .ihu-page { padding: 2.5rem; overflow-y: auto; } }
.ihu-head { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 0.75rem; }
.ihu-head-text { display: flex; flex-direction: column; gap: 0.4rem; min-width: 0; }
.ihu-head h3 { display: flex; align-items: center; gap: 0.5rem; margin: 0; font-size: 1.25rem; font-weight: 900; letter-spacing: -0.01em; }
.ihu-head p { margin: 0; font-size: 0.75rem; color: hsl(var(--muted-foreground)); }
.ihu-icon { width: 1.25rem; height: 1.25rem; color: hsl(var(--primary)); }
.ihu-file { width: fit-content; border-radius: 0.375rem; padding: 0.125rem 0.625rem; font-size: 0.75rem; font-weight: 600; background: hsl(var(--primary)); color: hsl(var(--primary-foreground)); }
.ihu-head-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
.ihu-btn { display: inline-flex; align-items: center; gap: 0.375rem; height: 2.25rem; padding: 0 0.75rem; border-radius: 0.5rem; border: 1px solid hsl(var(--input)); background: hsl(var(--background)); color: hsl(var(--foreground)); font-size: 0.8125rem; font-weight: 500; cursor: pointer; transition: background 0.15s; }
.ihu-btn:hover:not(:disabled) { background: hsl(var(--accent)); }
.ihu-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.ihu-btn-primary { border-color: transparent; background: hsl(var(--primary)); color: hsl(var(--primary-foreground)); }
.ihu-btn-primary:hover:not(:disabled) { background: hsl(var(--primary) / 0.9); }
.ihu-btn-icon { width: 0.875rem; height: 0.875rem; }
.ihu-spin { animation: ihu-spin 1s linear infinite; }
@keyframes ihu-spin { to { transform: rotate(360deg); } }
.ihu-alert { border-radius: 0.5rem; border: 1px solid hsl(var(--border)); background: hsl(var(--muted) / 0.4); padding: 0.625rem 0.875rem; font-size: 0.8125rem; }
.ihu-alert-error { border-color: hsl(var(--destructive) / 0.5); background: hsl(var(--destructive) / 0.08); color: hsl(var(--destructive)); }
.ihu-sections { display: flex; flex-direction: column; gap: 1.25rem; }
.ihu-sections label, .ihu-field { display: flex; flex-direction: column; gap: 0.375rem; min-width: 0; font-size: 0.8125rem; }
.ihu-sections label > span, .ihu-field > span { font-weight: 600; }
.ihu-sections small { font-size: 0.75rem; line-height: 1.25rem; color: hsl(var(--muted-foreground)); font-weight: 400; }
.ihu-wide { grid-column: 1 / -1; }
.ihu-sections input:not([type='checkbox']), .ihu-sections select, .ihu-sections textarea {
  width: 100%; box-sizing: border-box; border-radius: 0.5rem; border: 1px solid hsl(var(--input)); background: hsl(var(--background)); color: hsl(var(--foreground));
  padding: 0.5rem 0.75rem; font-size: 0.8125rem; font-family: inherit; outline: none; transition: border-color 0.15s, box-shadow 0.15s;
}
.ihu-sections input:not([type='checkbox']), .ihu-sections select { height: 2.25rem; }
.ihu-sections textarea { resize: vertical; min-height: 3.5rem; }
.ihu-sections input:focus, .ihu-sections select:focus, .ihu-sections textarea:focus { border-color: hsl(var(--ring)); box-shadow: 0 0 0 2px hsl(var(--ring) / 0.2); }
.ihu-sections select:disabled, .ihu-sections input:disabled { opacity: 0.55; }
.ihu-switch { flex-direction: row !important; align-items: flex-start; justify-content: space-between; gap: 1rem !important; border: 1px solid hsl(var(--border)); border-radius: 0.75rem; padding: 0.75rem 0.875rem; background: hsl(var(--muted) / 0.15); cursor: pointer; }
.ihu-switch > span { display: flex; flex-direction: column; gap: 0.25rem; font-weight: 400 !important; }
.ihu-switch b { font-weight: 600; }
.ihu-switch input { width: 1rem; height: 1rem; margin-top: 0.125rem; flex-shrink: 0; accent-color: hsl(var(--primary)); }
.ihu-chips { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; border: 1px solid hsl(var(--input)); border-radius: 0.5rem; padding: 0.375rem; background: hsl(var(--background)); }
.ihu-chip { display: inline-flex; align-items: center; gap: 0.25rem; border-radius: 0.375rem; background: hsl(var(--muted)); padding: 0.25rem 0.375rem 0.25rem 0.625rem; font-family: ui-monospace, monospace; font-size: 0.75rem; }
.ihu-chip button { display: inline-flex; border: 0; background: transparent; color: hsl(var(--muted-foreground)); cursor: pointer; padding: 0.125rem; border-radius: 0.25rem; }
.ihu-chip button:hover:not(:disabled) { color: hsl(var(--destructive)); }
.ihu-chip button:disabled { opacity: 0.35; cursor: not-allowed; }
.ihu-chip-x { width: 0.75rem; height: 0.75rem; }
.ihu-sections .ihu-chip-input { flex: 1; min-width: 9rem; height: 1.75rem !important; border: 0 !important; box-shadow: none !important; padding: 0 0.375rem !important; }
.ihu-foot { display: flex; justify-content: flex-end; }
.ihu-loading { display: flex; justify-content: center; padding: 5rem 0; }
.ihu-loading-icon { width: 2rem; height: 2rem; color: hsl(var(--primary) / 0.4); }
</style>
