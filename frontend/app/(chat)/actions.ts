"use server";

import type { UIMessage } from "ai";
import { cookies } from "next/headers";
import type { VisibilityType } from "@/components/visibility-selector";
import {
  deleteMessagesByChatIdAfterTimestamp,
  getChatById,
  getMessageById,
  saveChat,
  updateChatTitleById,
  updateChatVisiblityById,
} from "@/lib/db/queries";
import { getTextFromMessage } from "@/lib/utils";

export async function saveChatModelAsCookie(model: string) {
  const cookieStore = await cookies();
  cookieStore.set("chat-model", model);
}

export async function generateTitleFromUserMessage({
  message,
}: {
  message: UIMessage;
}) {
  const content = getTextFromMessage(message).trim();
  if (!content) {
    return "New Chat";
  }

  return buildFallbackTitle(message);
}

export async function ensureChatTitle({
  chatId,
  message,
  userId,
  visibility,
}: {
  chatId: string;
  message: UIMessage;
  userId: string;
  visibility: VisibilityType;
}) {
  let chat = await getChatById({ id: chatId });

  if (!chat) {
    await saveChat({
      id: chatId,
      userId,
      title: "New Chat",
      visibility,
    });
    chat = await getChatById({ id: chatId });
  }

  if (!chat) {
    return null;
  }

  if (chat.title && chat.title !== "New Chat") {
    return chat.title;
  }

  const generatedTitle = await generateTitleFromUserMessage({ message });
  const finalTitle = generatedTitle?.trim() || "New Chat";

  await updateChatTitleById({ chatId, title: finalTitle });

  return finalTitle;
}

function buildFallbackTitle(message: UIMessage) {
  const text = getTextFromMessage(message).trim();

  if (!text) {
    return "New Chat";
  }

  const sanitized = text
    .replace(/\s+/g, " ")
    .replace(/["“”'’`]/g, "")
    .trim();

  if (!sanitized) {
    return "New Chat";
  }

  const truncated =
    sanitized.length > 80 ? `${sanitized.slice(0, 77).trimEnd()}...` : sanitized;

  return truncated.charAt(0).toUpperCase() + truncated.slice(1);
}

export async function deleteTrailingMessages({ id }: { id: string }) {
  const [message] = await getMessageById({ id });

  await deleteMessagesByChatIdAfterTimestamp({
    chatId: message.chatId,
    timestamp: message.createdAt,
  });
}

export async function updateChatVisibility({
  chatId,
  visibility,
}: {
  chatId: string;
  visibility: VisibilityType;
}) {
  await updateChatVisiblityById({ chatId, visibility });
}
