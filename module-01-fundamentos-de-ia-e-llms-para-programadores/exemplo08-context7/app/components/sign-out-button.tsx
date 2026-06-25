"use client";

import { authClient } from "@/lib/auth-client";
import { useRouter } from "next/navigation";

export default function SignOutButton() {
  const router = useRouter();

  return (
    <button
      onClick={async () => {
        await authClient.signOut();
        router.refresh();
      }}
      className="px-4 py-2 bg-red-600 text-white font-medium rounded-lg hover:bg-red-500 transition-colors"
    >
      Sair
    </button>
  );
}
