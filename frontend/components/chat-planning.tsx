"use client";

import "@/styles/chat-planning.css";
import { useEffect, useRef, useState } from "react";
import type { ChatAttachment, ChatMessage } from "@/lib/chat/types";
import { ChatInlineContent } from "./chat-inline-input";
import { inlineChatUrls, normalizeChatUrl } from "@/lib/chat/inline-urls";
import { planningAnswers, type ChatPlanning } from "@/lib/chat/planning";
import { useChat } from "./chat-provider";

export function ChatWriterOffer() {
  const chat = useChat();
  return <>
    <p role="status" aria-atomic="true" className={chat.writerSeconds !== null || !chat.writerAnnouncement ? "chat-writer-announcement" : "chat-question-help"}>{chat.writerAnnouncement}</p>
    {chat.writerSeconds !== null && <aside className="chat-planning" aria-label="루트 작성으로 이동">
      <strong>지도와 함께 직관 계획을 이어가요</strong>
      <p className="chat-writer-countdown">{chat.writerSeconds}초 후 루트 작성 화면으로 자동 이동해요.</p>
      <p>이 대화와 작성 중인 내용은 유지돼요.</p>
      <div className="chat-question-actions">
        <button type="button" onClick={chat.goToWriter}>루트 작성으로 이동</button>
        <button type="button" onClick={() => chat.stayHere(true)}>여기 머무르기</button>
      </div>
    </aside>}
  </>;
}

export function ChatUserContent({ content, planning, attachments = [] }: { content: string; planning?: ChatPlanning; attachments?: ChatAttachment[] }) {
  const answers = planningAnswers(content, planning);
  const present = new Set(inlineChatUrls(content));
  const legacy = attachments.filter(item => {
    const url = item.kind === "url" && item.url ? normalizeChatUrl(item.url) : null;
    if (!url || present.has(url)) return false;
    present.add(url);
    return true;
  }).map(item => item.url).join("\n");
  const inline = <ChatInlineContent text={content + (legacy ? `\n${legacy}` : "")} attachments={attachments} />;
  if (!answers) return inline;
  return <><span>{answers.join(" · ")}</span><details className="chat-answer-details"><summary>답변 상세 보기</summary><div>{inline}</div></details></>;
}

export function ChatQuestions(props: { message: ChatMessage; disabled: boolean }) {
  if (!props.message.planning?.questions?.length || !props.message.id) return null;
  return <QuestionAnswers key={props.message.id} {...props} />;
}

function QuestionAnswers({ message, disabled }: { message: ChatMessage; disabled: boolean }) {
  const chat = useChat();
  const questions = message.planning!.questions;
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [direct, setDirect] = useState<Record<number, boolean>>({});
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [step, setStep] = useState(0);
  const [submission, setSubmission] = useState<"idle" | "pending" | "failed" | "rejected" | "succeeded">("idle");
  const sent = useRef(false);
  const heading = useRef<HTMLLegendElement>(null);
  const navigated = useRef(false);
  useEffect(() => { if (navigated.current) heading.current?.focus(); }, [step]);
  const locked = disabled || submission === "pending";
  const multiple = questions.length > 1;
  const question = questions[step];
  const complete = questions.every((_, i) => answers[i]?.trim());
  const submit = async (values: Record<number, string>) => {
    if (locked || sent.current || questions.some((_, i) => !values[i]?.trim())) return;
    sent.current = true;
    chat.stayHere();
    setAnswers(values);
    try {
      setSubmission(await chat.submitQuestions(message.id!, questions.map((q, i) => `${q.question}: ${values[i].trim()}`).join("\n"), () => setSubmission("pending")));
    } catch { setSubmission("failed"); }
    finally { sent.current = false; }
  };
  const move = (next: number) => {
    if (locked || sent.current) return;
    chat.stayHere();
    navigated.current = true;
    setStep(next);
  };
  const index = chat.messages.findIndex(item => item.id === message.id && item.role === "assistant");
  if (index >= 0 && index < chat.messages.length - 1) return null;
  if (submission === "pending" || (index >= 0 && chat.pending)) return <p className="chat-question-help chat-questions" role="status" aria-busy={true}>답변을 보내고 응답을 기다리고 있어요…</p>;
  return <div className="chat-planning chat-questions" aria-busy={false} onKeyDown={event => {
    if (event.key === "Enter" && event.target instanceof HTMLInputElement) {
      event.stopPropagation();
      if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229) return;
      event.preventDefault();
      if (event.repeat || locked || sent.current || !answers[step]?.trim()) return;
      if (multiple && step < questions.length - 1) move(step + 1);
      else void submit(answers);
    }
  }}>
    {multiple && <nav className="chat-question-progress" aria-label="질문 진행 상황">
      <span>{step + 1} / {questions.length}</span>
      <ol>{questions.map((q, i) => <li key={i} aria-current={i === step ? "step" : undefined} data-state={i === step ? "current" : answers[i]?.trim() ? "completed" : "pending"} aria-label={`${i + 1}. ${q.question} · ${i === step ? "현재" : answers[i]?.trim() ? "완료" : "대기"}`}>{i + 1}</li>)}</ol>
    </nav>}
    <fieldset disabled={locked}>
      <legend ref={heading} tabIndex={-1}>{question.question}</legend>
      {!multiple && question.choices.length > 0 && <p className="chat-question-help">선택하면 바로 보내요</p>}
      <div className="chat-question-choices">{question.choices.map(choice => <button type="button" key={choice} disabled={locked} aria-pressed={!direct[step] && answers[step] === choice} onClick={event => {
        event.preventDefault(); event.stopPropagation();
        if (locked || sent.current) return;
        chat.stayHere();
        const values = { ...answers, [step]: choice };
        setDirect(current => ({ ...current, [step]: false }));
        setAnswers(values);
        if (!multiple) submit(values);
      }}>{choice}</button>)}</div>
      <label className="chat-question-help" htmlFor={`planning-direct-${message.id}-${step}`}>직접 입력{direct[step] ? " · 선택됨" : drafts[step] ? " · 수정하면 이 답변을 선택해요" : ""}</label>
      <input id={`planning-direct-${message.id}-${step}`} aria-label={`${question.question} 직접 입력`} placeholder="답변을 직접 입력해주세요" maxLength={160} value={drafts[step] ?? ""} onChange={event => {
        if (locked || sent.current) return;
        chat.stayHere();
        setDirect(current => ({ ...current, [step]: true }));
        setDrafts(current => ({ ...current, [step]: event.target.value }));
        setAnswers(current => ({ ...current, [step]: event.target.value }));
      }} />
    </fieldset>
    <p className="chat-question-help">{multiple ? "보내기 전까지 답변을 바꿀 수 있어요. " : ""}채팅창으로 자유롭게 답해도 돼요.</p>
    <p role="status" className="chat-question-help">{submission === "failed" ? disabled ? "응답을 받지 못했어요. 저장된 질문은 채팅에서 수정하거나 다시 시도해 주세요." : "응답을 받지 못했어요. 답변을 바꾸거나 다시 보내 주세요." : submission === "rejected" ? "지금은 보낼 수 없어요. 연결 상태와 진행 중인 요청을 확인한 뒤 다시 보내 주세요." : ""}</p>
    {(multiple || !question.choices.length || direct[step]) && <div className="chat-question-actions">
      {multiple && <button type="button" disabled={locked || step === 0} onClick={() => move(step - 1)}>이전</button>}
      {multiple && step < questions.length - 1 ? <button type="button" className="chat-question-primary" disabled={locked || !answers[step]?.trim()} onClick={() => move(step + 1)}>다음</button> : <button type="button" className="chat-question-primary" disabled={locked || !complete} onClick={event => {
        event.preventDefault(); event.stopPropagation(); submit(answers);
      }}>답변 보내기</button>}
    </div>}
  </div>;
}
