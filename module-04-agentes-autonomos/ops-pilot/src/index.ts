import { createApp } from "./http/server.ts";

const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;

createApp().listen(PORT, () => {
  console.log(`OpsPilot ouvindo na porta ${PORT}`);
});
