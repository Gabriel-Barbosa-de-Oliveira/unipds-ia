import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import SignInButton from "./components/sign-in-button";
import SignOutButton from "./components/sign-out-button";

export default async function Home() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-6">
      <h1 className="text-4xl font-bold">Hello World</h1>

      {session ? (
        <div className="flex flex-col items-center gap-4">
          <p className="text-lg text-gray-600">
            Logado como{" "}
            <span className="font-semibold text-gray-900">
              {session.user.name || session.user.email}
            </span>
          </p>
          <SignOutButton />
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4">
          <p className="text-lg text-gray-500">Você não está logado</p>
          <SignInButton />
        </div>
      )}
    </main>
  );
}
