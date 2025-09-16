# Your AI Chatbot Repo — Explained in Simple Words

_This document labels every folder and file and explains what it’s for — in plain English._

## Big Picture
- **app/**: the website and API.
- **components/**: reusable visual pieces.
- **lib/**: backend helpers (AI, database, editor, tools).
- **artifacts/**: panels for AI-created things (code/text/image/sheet).
- **hooks/**: small helpers for UI behavior.
- **public/**: images/icons the browser can fetch directly.
- **tests/**: automated checks.
- Plus config files that tie it all together.


**Your change (xAI → OpenAI):** look at `lib/ai/providers.ts` (and possibly `instrumentation.ts`). That’s where the app picks which AI provider/models to use. Set `OPENAI_API_KEY` in `.env.local`.

## app/

- `app/(auth)/actions.ts` — Server actions (server-side functions) used by **/**.
- `app/(auth)/api/auth/[...nextauth]/route.ts` — Auth.js (NextAuth) API endpoint (login/session callbacks).
- `app/(auth)/api/auth/guest/route.ts` — API endpoint that handles requests at **/api/auth/guest/route.ts**.
- `app/(auth)/auth.config.ts` — Part of the app route at **/auth.config.ts**.
- `app/(auth)/auth.ts` — Part of the app route at **/auth.ts**.
- `app/(auth)/login/page.tsx` — The actual page people see at **/login**.
- `app/(auth)/register/page.tsx` — The actual page people see at **/register**.
- `app/(chat)/actions.ts` — Server actions (server-side functions) used by **/**.
- `app/(chat)/api/chat/[id]/stream/route.ts` — Streaming API endpoint for **/api/chat/:id/stream** (sends tokens as they’re generated).
- `app/(chat)/api/chat/route.ts` — API endpoint that handles requests at **/api/chat/route.ts**.
- `app/(chat)/api/chat/schema.ts` — Validation rules/types used by the API at **/api/chat**.
- `app/(chat)/api/document/route.ts` — API endpoint that handles requests at **/api/document/route.ts**.
- `app/(chat)/api/files/upload/route.ts` — API endpoint that handles requests at **/api/files/upload/route.ts**.
- `app/(chat)/api/history/route.ts` — API endpoint that handles requests at **/api/history/route.ts**.
- `app/(chat)/api/suggestions/route.ts` — API endpoint that handles requests at **/api/suggestions/route.ts**.
- `app/(chat)/api/vote/route.ts` — API endpoint that handles requests at **/api/vote/route.ts**.
- `app/(chat)/chat/[id]/page.tsx` — The actual page people see at **/chat/:id**.
- `app/(chat)/layout.tsx` — The page frame for **/** (shared header/sidebars).
- `app/(chat)/opengraph-image.png` — Part of the app route at **/opengraph-image.png**.
- `app/(chat)/page.tsx` — The actual page people see at **/**.
- `app/(chat)/twitter-image.png` — Part of the app route at **/twitter-image.png**.
- `app/favicon.ico` — The little browser tab icon.
- `app/globals.css` — Global CSS styles for the whole site.
- `app/layout.tsx` — The page frame for **/** (shared header/sidebars).

## components/

- `components/app-sidebar.tsx` — The left sidebar parts (history, buttons, toggles).
- `components/artifact-actions.tsx` — Shows or controls an AI ‘artifact’ (code/text/image/sheet panel).
- `components/artifact-close-button.tsx` — Shows or controls an AI ‘artifact’ (code/text/image/sheet panel).
- `components/artifact-messages.tsx` — Shows a single chat message or a list of them.
- `components/artifact.tsx` — Shows or controls an AI ‘artifact’ (code/text/image/sheet panel).
- `components/auth-form.tsx` — The log-in / sign-up form component.
- `components/chat-header.tsx` — The top bar for the chat screen.
- `components/chat.tsx` — The main chat interface (message list + input).
- `components/code-editor.tsx` — Code editor used inside artifacts.
- `components/console.tsx` — Developer console panel for diagnostics.
- `components/create-artifact.tsx` — Shows or controls an AI ‘artifact’ (code/text/image/sheet panel).
- `components/data-stream-handler.tsx` — A reusable UI piece for the chat app.
- `components/data-stream-provider.tsx` — A reusable UI piece for the chat app.
- `components/diffview.tsx` — A reusable UI piece for the chat app.
- `components/document-preview.tsx` — Document preview or skeleton loader.
- `components/document-skeleton.tsx` — Document preview or skeleton loader.
- `components/document.tsx` — Document preview or skeleton loader.
- `components/elements/actions.tsx` — A reusable UI piece for the chat app.
- `components/elements/branch.tsx` — A reusable UI piece for the chat app.
- `components/elements/code-block.tsx` — A reusable UI piece for the chat app.
- `components/elements/context.tsx` — A reusable UI piece for the chat app.
- `components/elements/conversation.tsx` — A reusable UI piece for the chat app.
- `components/elements/image.tsx` — A reusable UI piece for the chat app.
- `components/elements/inline-citation.tsx` — A reusable UI piece for the chat app.
- `components/elements/loader.tsx` — A reusable UI piece for the chat app.
- `components/elements/message.tsx` — Shows a single chat message or a list of them.
- `components/elements/prompt-input.tsx` — A reusable UI piece for the chat app.
- `components/elements/reasoning.tsx` — A reusable UI piece for the chat app.
- `components/elements/response.tsx` — A reusable UI piece for the chat app.
- `components/elements/source.tsx` — A reusable UI piece for the chat app.
- `components/elements/suggestion.tsx` — Shows suggested follow-up actions/prompts.
- `components/elements/task.tsx` — A reusable UI piece for the chat app.
- `components/elements/tool.tsx` — A reusable UI piece for the chat app.
- `components/elements/web-preview.tsx` — A reusable UI piece for the chat app.
- `components/greeting.tsx` — A reusable UI piece for the chat app.
- `components/icons.tsx` — A reusable UI piece for the chat app.
- `components/image-editor.tsx` — A reusable UI piece for the chat app.
- `components/message-actions.tsx` — Shows a single chat message or a list of them.
- `components/message-editor.tsx` — Shows a single chat message or a list of them.
- `components/message-reasoning.tsx` — Shows a single chat message or a list of them.
- `components/message.tsx` — Shows a single chat message or a list of them.
- `components/messages.tsx` — Shows a single chat message or a list of them.
- `components/model-selector.tsx` — A reusable UI piece for the chat app.
- `components/multimodal-input.tsx` — A reusable UI piece for the chat app.
- `components/preview-attachment.tsx` — A reusable UI piece for the chat app.
- `components/sheet-editor.tsx` — A reusable UI piece for the chat app.
- `components/sidebar-history-item.tsx` — The left sidebar parts (history, buttons, toggles).
- `components/sidebar-history.tsx` — The left sidebar parts (history, buttons, toggles).
- `components/sidebar-toggle.tsx` — The left sidebar parts (history, buttons, toggles).
- `components/sidebar-user-nav.tsx` — The left sidebar parts (history, buttons, toggles).
- `components/sign-out-form.tsx` — A reusable UI piece for the chat app.
- `components/submit-button.tsx` — A reusable UI piece for the chat app.
- `components/suggested-actions.tsx` — Shows suggested follow-up actions/prompts.
- `components/suggestion.tsx` — Shows suggested follow-up actions/prompts.
- `components/text-editor.tsx` — Text editor used inside artifacts.
- `components/theme-provider.tsx` — Sets light/dark theme for the app.
- `components/toast.tsx` — Small popup notification.
- `components/toolbar.tsx` — A reusable UI piece for the chat app.
- `components/ui/alert-dialog.tsx` — A reusable UI piece for the chat app.
- `components/ui/avatar.tsx` — A reusable UI piece for the chat app.
- `components/ui/badge.tsx` — A reusable UI piece for the chat app.
- `components/ui/button.tsx` — A reusable UI piece for the chat app.
- `components/ui/card.tsx` — A reusable UI piece for the chat app.
- `components/ui/carousel.tsx` — A reusable UI piece for the chat app.
- `components/ui/collapsible.tsx` — A reusable UI piece for the chat app.
- `components/ui/dropdown-menu.tsx` — A reusable UI piece for the chat app.
- `components/ui/hover-card.tsx` — A reusable UI piece for the chat app.
- `components/ui/input.tsx` — A reusable UI piece for the chat app.
- `components/ui/label.tsx` — A reusable UI piece for the chat app.
- `components/ui/progress.tsx` — A reusable UI piece for the chat app.
- `components/ui/scroll-area.tsx` — A reusable UI piece for the chat app.
- `components/ui/select.tsx` — A reusable UI piece for the chat app.
- `components/ui/separator.tsx` — A reusable UI piece for the chat app.
- `components/ui/sheet.tsx` — A reusable UI piece for the chat app.
- `components/ui/sidebar.tsx` — The left sidebar parts (history, buttons, toggles).
- `components/ui/skeleton.tsx` — A reusable UI piece for the chat app.
- `components/ui/textarea.tsx` — A reusable UI piece for the chat app.
- `components/ui/tooltip.tsx` — A reusable UI piece for the chat app.
- `components/version-footer.tsx` — A reusable UI piece for the chat app.
- `components/visibility-selector.tsx` — A reusable UI piece for the chat app.
- `components/weather.tsx` — Small widget to display weather (used by the weather tool).

## lib/

- `lib/ai/entitlements.ts` — Feature flags/limits (who can use what).
- `lib/ai/models.mock.ts` — Mock models used in tests.
- `lib/ai/models.test.ts` — Tests around model selection.
- `lib/ai/models.ts` — List & config of AI model options used in the app.
- `lib/ai/prompts.ts` — System prompts and starter instructions for the AI.
- `lib/ai/providers.ts` — Sets which AI provider/models to use (where you switched xAI → OpenAI).
- `lib/ai/tools/create-document.ts` — Tool to create a new document artifact.
- `lib/ai/tools/get-weather.ts` — Tool the AI can call to fetch the weather.
- `lib/ai/tools/request-suggestions.ts` — Tool to generate suggested follow-up actions.
- `lib/ai/tools/update-document.ts` — Tool to update an existing document artifact.
- `lib/artifacts/server.ts` — Server-side helpers for artifact streaming & storage.
- `lib/constants.ts` — Shared constants used across the app.
- `lib/db/helpers/01-core-to-parts.ts` — Helper code used by DB scripts.
- `lib/db/migrate.ts` — Script/helper to run database migrations.
- `lib/db/migrations/0000_keen_devos.sql` — A database migration file (how to change DB over time).
- `lib/db/migrations/0001_sparkling_blue_marvel.sql` — A database migration file (how to change DB over time).
- `lib/db/migrations/0002_wandering_riptide.sql` — A database migration file (how to change DB over time).
- `lib/db/migrations/0003_cloudy_glorian.sql` — A database migration file (how to change DB over time).
- `lib/db/migrations/0004_odd_slayback.sql` — A database migration file (how to change DB over time).
- `lib/db/migrations/0005_wooden_whistler.sql` — A database migration file (how to change DB over time).
- `lib/db/migrations/0006_marvelous_frog_thor.sql` — A database migration file (how to change DB over time).
- `lib/db/migrations/0007_flowery_ben_parker.sql` — A database migration file (how to change DB over time).
- `lib/db/migrations/meta/0000_snapshot.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/migrations/meta/0001_snapshot.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/migrations/meta/0002_snapshot.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/migrations/meta/0003_snapshot.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/migrations/meta/0004_snapshot.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/migrations/meta/0005_snapshot.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/migrations/meta/0006_snapshot.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/migrations/meta/0007_snapshot.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/migrations/meta/_journal.json` — Drizzle’s metadata for migrations (snapshots/journal).
- `lib/db/queries.ts` — Reusable DB queries for reading/writing chat data.
- `lib/db/schema.ts` — Database tables (chat sessions, messages, etc.).
- `lib/db/utils.ts` — Small DB utilities (connections, helpers).
- `lib/editor/config.ts` — Settings for the in-app editors.
- `lib/editor/diff.js` — Code to show differences between two versions of text/code.
- `lib/editor/functions.tsx` — Rendering and helper logic for editor suggestions.
- `lib/editor/react-renderer.tsx` — Rendering and helper logic for editor suggestions.
- `lib/editor/suggestions.tsx` — Rendering and helper logic for editor suggestions.
- `lib/errors.ts` — Common error types and helpers.
- `lib/types.ts` — Shared TypeScript types for strong typing.
- `lib/usage.ts` — Tracks token usage/costs for analytics/limits.
- `lib/utils.ts` — General-purpose helpers used everywhere.

## artifacts/

- `artifacts/actions.ts` — Server actions for creating/saving artifacts.
- `artifacts/code/client.tsx` — Client-side viewer/editor for a code artifact.
- `artifacts/code/server.ts` — Server-side logic for code artifacts.
- `artifacts/image/client.tsx` — Component to show an image artifact.
- `artifacts/sheet/client.tsx` — Components to show/edit a spreadsheet-like artifact.
- `artifacts/sheet/server.ts` — Components to show/edit a spreadsheet-like artifact.
- `artifacts/text/client.tsx` — Components to show/edit a text document artifact.
- `artifacts/text/server.ts` — Components to show/edit a text document artifact.

## hooks/

- `hooks/use-artifact.ts` — Manages currently open artifact (code/text/sheet/image).
- `hooks/use-auto-resume.ts` — Resumes streaming after refresh/network hiccup.
- `hooks/use-chat-visibility.ts` — Shows/hides chat panel (e.g., with artifacts).
- `hooks/use-messages.tsx` — React hook to manage chat messages (state & updates).
- `hooks/use-mobile.tsx` — Detects mobile layout to tweak UI.
- `hooks/use-scroll-to-bottom.tsx` — Keeps the chat scrolled to the latest message.

## public/

- `public/images/demo-thumbnail.png` — Static image used in previews/demos.
- `public/images/mouth of the seine, monet.jpg` — Static image used in previews/demos.

## tests/

- `tests/e2e/artifacts.test.ts` — Project file.
- `tests/e2e/chat.test.ts` — Project file.
- `tests/e2e/reasoning.test.ts` — Project file.
- `tests/e2e/session.test.ts` — Project file.
- `tests/fixtures.ts` — Project file.
- `tests/helpers.ts` — Project file.
- `tests/pages/artifact.ts` — Project file.
- `tests/pages/auth.ts` — Project file.
- `tests/pages/chat.ts` — Project file.
- `tests/prompts/basic.ts` — Project file.
- `tests/prompts/routes.ts` — Project file.
- `tests/prompts/utils.ts` — Project file.
- `tests/routes/chat.test.ts` — Project file.
- `tests/routes/document.test.ts` — Project file.

## .github/

- `.github/workflows/lint.yml` — CI workflow that runs linting on pushes/PRs.
- `.github/workflows/playwright.yml` — CI workflow that runs Playwright tests.

## .vscode/

- `.vscode/extensions.json` — Suggested VS Code extensions for this project.
- `.vscode/settings.json` — VS Code settings for consistent formatting.

## README.md/

- `README.md` — Intro and instructions for the project.

## LICENSE/

- `LICENSE` — The legal license for this code.

## .gitignore/

- `.gitignore` — Tells Git which files/folders to ignore (like node_modules, .env).

## .env.example/

- `.env.example` — A sample of all the secret settings the app needs (copy to .env.local and fill).

## .env.local/

- `.env.local` — Your private secrets for running locally (never commit this).

## drizzle.config.ts/

- `drizzle.config.ts` — Drizzle ORM setup (how to connect to Postgres & run migrations).

## instrumentation.ts/

- `instrumentation.ts` — Runs before the app starts; good place to set default AI provider.

## middleware.ts/

- `middleware.ts` — Code that runs on every request (e.g., auth redirects).

## package.json/

- `package.json` — Project name, scripts (dev/build/test), and dependencies.

## tsconfig.json/

- `tsconfig.json` — TypeScript compiler settings.

## next.config.ts/

- `next.config.ts` — Build/runtime settings for Next.js.

## next-env.d.ts/

- `next-env.d.ts` — TypeScript helpers for Next.js.

## postcss.config.mjs/

- `postcss.config.mjs` — CSS processing config (used with Tailwind).

## components.json/

- `components.json` — Settings for shadcn/ui (design system generator).

## biome.jsonc/

- `biome.jsonc` — Settings for Biome (a fast formatter/linter).

## playwright.config.ts/

- `playwright.config.ts` — Playwright test runner settings.

## pnpm-lock.yaml/

- `pnpm-lock.yaml` — Exact versions of installed packages (for reproducible installs).

## vercel-template.json/

- `vercel-template.json` — Metadata used by Vercel’s one-click deploy template.
