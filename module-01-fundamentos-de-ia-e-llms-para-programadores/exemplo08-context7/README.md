# Demo: Next.js + Better Auth + GitHub + SQLite

Demo mínimo de autenticação com GitHub OAuth, banco SQLite local e UI em Tailwind CSS.

## Pré-requisitos

- Node.js 18+
- Conta GitHub com OAuth App configurada

## Configurar OAuth App no GitHub

1. Acesse: **GitHub → Settings → Developer settings → OAuth Apps → New OAuth App**
2. Preencha:
   - **Application name**: qualquer nome
   - **Homepage URL**: `http://localhost:3000`
   - **Authorization callback URL**: `http://localhost:3000/api/auth/callback/github`
3. Copie o **Client ID** e gere um **Client Secret**

## Configurar variáveis de ambiente

Edite o arquivo `.env.local` na raiz do projeto:

```env
BETTER_AUTH_SECRET=um-segredo-aleatorio-longo-e-seguro
BETTER_AUTH_URL=http://localhost:3000
GITHUB_CLIENT_ID=seu-client-id-do-github
GITHUB_CLIENT_SECRET=seu-client-secret-do-github
```

## Rodar o projeto

```bash
# 1. Instalar dependências
npm install

# 2. Gerar tabelas do banco SQLite
npx @better-auth/cli migrate

# 3. Iniciar o servidor de desenvolvimento
npm run dev
```

Acesse [http://localhost:3000](http://localhost:3000)

## Estrutura de arquivos

```
lib/auth.ts                          # Configuração do servidor Better Auth
lib/auth-client.ts                   # Cliente para componentes React
app/api/auth/[...all]/route.ts       # Route handler Next.js
app/page.tsx                         # Página Home (server component)
app/components/sign-in-button.tsx    # Botão "Entrar com GitHub"
app/components/sign-out-button.tsx   # Botão "Sair"
```
