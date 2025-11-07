import { mount } from '@vue/test-utils'
import { describe, it, expect } from 'vitest'
import TranslationBox from '../TranslationBox.vue'

describe('TranslationBox.vue', () => {
  it('renders title and default translation message', () => {
    const wrapper = mount(TranslationBox)
    expect(wrapper.text()).toContain('Translation:')
    expect(wrapper.text()).toContain('Waiting for sign input...')
  })
})
