import { cookies } from "next/headers";
import { notFound, redirect } from "next/navigation";

import { auth } from "@/app/(auth)/auth";
import { Chat } from "@/components/chat";
import { DataStreamHandler } from "@/components/data-stream-handler";
import { DEFAULT_CHAT_MODEL } from "@/lib/ai/models";
import { getChatById, getMessagesByChatId } from "@/lib/db/queries";
import { FASTAPI_HISTORY_ENDPOINT } from "@/lib/config";
import {
  convertMicroserviceHistoryToChatMessages,
  convertToUIMessages,
  type MicroserviceHistoryResponse,
} from "@/lib/utils";

export default async function Page(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  const { id } = params;
  const session = await auth();

  if (!session) {
    redirect("/api/auth/guest");
  }

  const chat = await getChatById({ id });

  if (!chat) {
    notFound();
  }

  if (chat.visibility === "private") {
    if (!session.user) {
      return notFound();
    }

    if (session.user.id !== chat.userId) {
      return notFound();
    }
  }

  const messagesFromDb = await getMessagesByChatId({
    id,
  });

  let uiMessages = convertToUIMessages(messagesFromDb);

  try {
    const historyResponse = await fetch(FASTAPI_HISTORY_ENDPOINT, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ thread_id: id }),
      cache: "no-store",
    });

    if (historyResponse.ok) {
      const historyJson =
        (await historyResponse.json()) as MicroserviceHistoryResponse;
      const microserviceMessages =
        convertMicroserviceHistoryToChatMessages(historyJson);

      if (microserviceMessages.length > 0) {
        uiMessages = microserviceMessages;
      }
    } else {
      console.warn("Microservice history request failed", {
        chatId: id,
        status: historyResponse.status,
      });
    }
  } catch (error) {
    console.warn("Microservice history request error", {
      chatId: id,
      error,
    });
  }

  const cookieStore = await cookies();
  const chatModelFromCookie = cookieStore.get("chat-model");

  if (!chatModelFromCookie) {
    return (
      <>
        <Chat
          autoResume={true}
          id={chat.id}
          initialChatModel={DEFAULT_CHAT_MODEL}
          initialLastContext={chat.lastContext ?? undefined}
          initialMessages={uiMessages}
          initialTitle={chat.title}
          initialVisibilityType={chat.visibility}
          isReadonly={session?.user?.id !== chat.userId}
          userId={session?.user?.id ?? ""}
        />
        <DataStreamHandler />
      </>
    );
  }

  return (
    <>
      <Chat
        autoResume={true}
        id={chat.id}
        initialChatModel={chatModelFromCookie.value}
        initialLastContext={chat.lastContext ?? undefined}
        initialMessages={uiMessages}
        initialTitle={chat.title}
        initialVisibilityType={chat.visibility}
        isReadonly={session?.user?.id !== chat.userId}
        userId={session?.user?.id ?? ""}
      />
      <DataStreamHandler />
    </>
  );
}
