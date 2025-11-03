import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import tsEslint from 'typescript-eslint'
import eslintPluginPrettierRecommended from '@vue/eslint-config-prettier'

export default [
  {
    ignores: ['node_modules', 'dist'],
  },
  js.configs.recommended,
  ...tsEslint.configs.recommended,
  ...pluginVue.configs['flat/essential'],
  eslintPluginPrettierRecommended,
  {
    files: ['src/**/*.{js,jsx,ts,tsx,vue}'],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
    },
    rules: {
      'vue/multi-word-component-names': 'off',
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
    },
  },
  {
    files: ['src/**/*.vue'],
    languageOptions: {
      parser: pluginVue.parser,
      parserOptions: {
        parser: tsEslint.parser,
        sourceType: 'module',
        ecmaVersion: 2020,
      },
    },
  },
]
