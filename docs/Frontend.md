# Documentation for frontend
## Overview
- **Location**: `web/` — a small frontend application built with Vue 3 and Vite.
- **Purpose**: Provide developers a quick start guide, run/build instructions, technical highlights, folder map, and common troubleshooting notes.

## Quick Start
- **Prerequisites**: Node.js (recommended 18+), and a package manager (`pnpm` recommended, or `npm` / `yarn`).
- **Install dependencies**: Run in the `web/` directory (recommended):

```bash
cd web
pnpm install
# or with npm:
# npm install
```

- **Run locally**: Start the dev server with hot reload:

```bash
cd web
pnpm dev
# or: pnpm run dev
```

- **Build & preview**:

```bash
cd web
pnpm build       # runs type check (vue-tsc) then vite build
pnpm preview     # preview the built output locally
```

## Scripts (from `web/package.json`)
- `dev`: `vite` — start dev server.
- `build`: `vue-tsc && vite build` — run type-check then build.
- `preview`: `vite preview` — preview the production build.
- `format`: `prettier --write src/**/*.{vue,ts,tsx,js,jsx,json,css,scss}` — format source files.
- `lint`: `eslint src --ext .ts,.tsx,.js,.jsx,.vue` — run ESLint checks.

(Use `pnpm run <script>` or `npm run <script>`.)

## Technical highlights
- **Framework**: Vue 3 (Composition API).
- **Bundler / Dev tool**: Vite for fast cold starts and HMR.
- **Type checking**: TypeScript + `vue-tsc` (executed during `build`).
- **Routing**: `vue-router` (routes are defined in `web/src/router/index.ts`).
- **HTTP client**: `axios` for calls to backend / agent APIs.
- **Markdown processing**: `markdown-it` and `markdown-it-anchor` for rendering and anchors.
- **Lint & formatting**: ESLint + Prettier with `@vue/eslint-config-typescript` and `@vue/eslint-config-prettier`.

## Project structure (brief)
- `web/`
  - `index.html` — Vite entry template.
  - `package.json` — scripts and dependencies.
  - `src/`
    - `main.ts` — app bootstrap (mounts Vue app and router).
    - `App.vue` — root component.
    - `style.css` — global CSS.
    - `router/`
      - `index.ts` — route definitions (`Home`, `RepoDetail`, ...).
    - `pages/`
      - `Home.vue` — home page.
      - `RepoDetail.vue` — repository detail page.
    - `components/`
      - `Header.vue` — top navigation / header.
      - `InfoCard.vue` — card for displaying lists/info.
      - `InputSection.vue` — input / query area.
      - `ThemeToggle.vue` — theme toggle control.
    - `utils/` — shared frontend utilities (e.g. API request helpers).

## Key files
- `web/src/main.ts`: bootstraps the app, registers router and any global plugins.
- `web/src/router/index.ts`: contains route table and route-level configuration.
- `web/src/pages/Home.vue`: the home page — primary entry UI.
- `web/src/pages/RepoDetail.vue`: repository detail and analysis view.
- `web/src/components/*`: reusable UI components following single-responsibility.
- `web/style.css`: global CSS variables and base styles.

## Development practices & tools
- Use `pnpm format` (Prettier) to format code consistently.
- Use `pnpm lint` (ESLint) to run static analysis and fix issues early.
- It's recommended to run `pnpm format` and `pnpm lint` before committing.
- Run frontend tests (if present) and ensure type checks pass before `build`.

## Debugging & common issues
- Dev server won't start:
  - Verify Node.js version (recommend >= 18).
  - Make sure dependencies are installed in `web/` (`pnpm install`).
  - If the default port is in use, Vite will suggest an alternative or you can set `PORT`

- ESLint errors/warnings: fix according to rules, or temporarily disable a rule with a comment; consider updating `.eslintrc` to align with team conventions.
- Type errors (`vue-tsc`): follow the error messages and correct type mismatches; avoid using `any` unless necessary.

## Build & deployment suggestions
- Build with `pnpm build` — output goes to Vite's `dist/` folder by default.
- Host static output on services like Vercel, Netlify, GitHub Pages, or serve via Nginx/static server.
- If deploying to a sub-path, set `base` in `vite.config.ts` accordingly.

## Optional enhancements & notes
- For SSR or advanced prefetching, consider Nuxt or a Vite SSR setup.
- For local API proxying, configure `server.proxy` in `vite.config.ts` (e.g. forward `/api` to backend).
- If the project grows into a monorepo or micro-frontends, extract shared components into a separate package.

## Contributing & contact
- To contribute, create a branch, submit a PR, and follow the project's code style and testing requirements.
- For questions, open an issue in the repository or contact the maintainers directly.