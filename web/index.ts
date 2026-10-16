import UpscaleSettings from './views/UpscaleSettings.vue'
import { host } from './host'

host.registerPlugin({
  id: 'image_hd_upscale',
  configEditors: [
    {
      files: ['image_hd_upscale.yaml', 'image_hd_upscale.yml'],
      title: '图片放大',
      component: UpscaleSettings,
    },
  ],
})
