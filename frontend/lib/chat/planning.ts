export type ChatPlanning = { offer_writer: boolean; questions: { question: string; choices: string[] }[] };
// Only the immediately originating structured questions can identify a combined answer.
export function planningAnswers(content: string, planning?: ChatPlanning): string[] | undefined {
  if (!planning?.questions.length) return;
  let rest = content;
  const answers: string[] = [];
  for (let i = 0; i < planning.questions.length; i++) {
    const prefix = `${planning.questions[i].question}: `;
    if (!rest.startsWith(prefix)) return;
    rest = rest.slice(prefix.length);
    const next = planning.questions[i + 1];
    const boundary = next ? rest.indexOf(`\n${next.question}: `) : rest.length;
    if (boundary < 0) return;
    const answer = rest.slice(0, boundary);
    if (!answer.trim()) return;
    answers.push(answer);
    rest = next ? rest.slice(boundary + 1) : "";
  }
  return answers;
}

export function parsePlanning(value: unknown): ChatPlanning | undefined {
  if (!value || typeof value !== "object") return;
  const payload = value as ChatPlanning;
  if (typeof payload.offer_writer !== "boolean" || !Array.isArray(payload.questions) || payload.questions.length > 4 || (!payload.questions.length && !payload.offer_writer)) return;
  for (const q of payload.questions) {
    if (!q || typeof q.question !== "string" || !q.question.trim() || q.question.length > 160 || !Array.isArray(q.choices) || q.choices.length === 1 || q.choices.length > 10 ||
      q.choices.some(c => typeof c !== "string" || !c.trim() || c.length > 80) || new Set(q.choices).size !== q.choices.length) return;
  }
  return { offer_writer: payload.offer_writer, questions: payload.questions.map(q => ({ question: q.question, choices: [...q.choices] })) };
}
