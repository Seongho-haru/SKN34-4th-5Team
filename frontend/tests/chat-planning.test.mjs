import assert from "node:assert/strict";
import { test } from "node:test";
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const frontend = dirname(dirname(fileURLToPath(import.meta.url)));
const nodes = node => !node || typeof node !== "object" ? [] : Array.isArray(node) ? node.flatMap(nodes) : [node, ...nodes(node.props?.children)];

async function harness(count = 2, run) {
  const scratch = mkdtempSync(join(tmpdir(), "kbo-planning-"));
  let slots = [], cursor = 0, key;
  const sends = [], effects = [];
  let stays = 0;
  const require = createRequire(join(scratch, "entry.cjs"));
  try {
    symlinkSync(join(frontend, "node_modules"), join(scratch, "node_modules"), "dir");
    writeFileSync(join(scratch, "hooks.cjs"), "module.exports = global.__planningHooks;");
    global.__planningHooks = {
      useState(initial) { const i = cursor++; if (!(i in slots)) slots[i] = initial; return [slots[i], value => { slots[i] = typeof value === "function" ? value(slots[i]) : value; }]; },
      useRef(initial) { const i = cursor++; return slots[i] ??= { current: initial }; },
      useEffect(fn, deps) { const i = cursor++; if (!slots[i] || deps.some((v, j) => v !== slots[i][j])) effects.push(fn); slots[i] = deps; },
      useChat: () => ({ messages: [], pending: "", stayHere() { stays++; }, submitQuestions: (id, text, accepted) => { sends.push([id, text]); accepted?.(); return Promise.resolve("succeeded"); } }),
    };
    const source = readFileSync(join(frontend, "components/chat-planning.tsx"), "utf8")
      .replace(/^import "@\/styles\/[^\"]+";$/m, "")
      .replace('from "react"', 'from "./hooks.cjs"')
      .replace('from "./chat-provider"', 'from "./hooks.cjs"')
      .replace('from "./chat-inline-input"', 'from "./inline.cjs"')
      .replace('from "@/lib/chat/inline-urls"', 'from "./inline.cjs"')
      .replace('from "@/lib/chat/planning"', 'from "./helper.cjs"');
    writeFileSync(join(scratch, "inline.cjs"), ts.transpileModule(readFileSync(join(frontend, "lib/chat/inline-urls.ts"), "utf8"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText + "\nexports.ChatInlineContent = () => null;");
    writeFileSync(join(scratch, "helper.cjs"), ts.transpileModule(readFileSync(join(frontend, "lib/chat/planning.ts"), "utf8"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText);
    writeFileSync(join(scratch, "planning.cjs"), ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText);
    const { ChatQuestions } = require("./planning.cjs");
    const props = { message: { id: 2, role: "assistant", planning: { questions: Array.from({ length: count }, (_, i) => ({ question: `질문 ${i + 1}`, choices: ["혼자", "친구", "가족", "동료"] })) } }, disabled: false };
    let tree, focuses = 0;
    const render = () => {
      const wrapper = ChatQuestions(props);
      if (!wrapper) return tree = null;
      if (key !== wrapper.key) { key = wrapper.key; slots = []; }
      cursor = 0;
      tree = wrapper.type(wrapper.props);
      const legend = nodes(tree).find(n => n.type === "legend");
      if (legend) legend.props.ref.current = { focus() { focuses++; } };
      effects.splice(0).forEach(fn => fn());
      return tree;
    };
    const button = text => nodes(tree).find(n => n.type === "button" && [n.props.children].flat().join("") === text);
    const click = node => {
      assert.ok(node, "button exists");
      assert.equal(node.props.type, "button");
      if (!node.props.disabled) node.props.onClick({ preventDefault() {}, stopPropagation() {} });
      render();
    };
    render();
    return await run({ props, render, button, click, sends, get tree() { return tree; }, get stays() { return stays; }, get focuses() { return focuses; }, require });
  } finally { delete global.__planningHooks; rmSync(scratch, { recursive: true, force: true }); }
}

test("writer offer ticks only visible sentence, keeps names and persistent one-time status in both surfaces", () => harness(1, h => {
  const { ChatWriterOffer } = h.require("./planning.cjs");
  let explicit;
  const chat = { writerSeconds: null, writerAnnouncement: "", goToWriter() {}, stayHere(value) { explicit = value; } };
  global.__planningHooks.useChat = () => chat;
  const render = () => ChatWriterOffer();
  const status = tree => nodes(tree).find(n => n.props?.role === "status");
  assert.equal(status(render()).props.children, "");
  chat.writerSeconds = 20;
  chat.writerAnnouncement = "20초 후 루트 작성 화면으로 자동 이동해요. 여기 머무르기를 누르면 자동 이동을 취소할 수 있어요.";
  for (const seconds of [20, 19, 1]) {
    chat.writerSeconds = seconds;
    const tree = render();
    assert.equal(status(tree).props.children, chat.writerAnnouncement);
    assert.equal(status(tree).props["aria-atomic"], "true");
    assert.deepEqual(nodes(tree).filter(n => n.type === "button").map(n => n.props.children), ["루트 작성으로 이동", "여기 머무르기"]);
    assert.equal(nodes(tree).find(n => n.props?.className === "chat-writer-countdown").props.children.join(""), `${seconds}초 후 루트 작성 화면으로 자동 이동해요.`);
    assert.equal(nodes(tree).find(n => n.type === "aside").props["aria-live"], undefined);
    nodes(tree).find(n => n.type === "button" && n.props.children === "여기 머무르기").props.onClick();
    assert.equal(explicit, true);
  }
  chat.writerSeconds = null; chat.writerAnnouncement = "자동 이동을 취소했어요. 여기서 대화를 이어가요.";
  const tree = render();
  assert.equal(status(tree).props.children, chat.writerAnnouncement);
  assert.equal(status(tree).props.className, "chat-question-help");
  assert.equal(nodes(tree).some(n => n.type === "aside"), false);
  for (const surface of ["chat-workspace", "chat-popup"]) assert.match(readFileSync(join(frontend, `components/${surface}.tsx`), "utf8"), /<ChatWriterOffer \/>/);
}));

for (const count of [1, 2]) test(`accepted ${count}-question group collapses; persisted follow-up hides it and deletion reopens same state`, () => harness(count, async h => {
  let messages = [h.props.message], pending = "";
  global.__planningHooks.useChat = () => ({ messages, pending, stayHere() {}, submitQuestions: (_id, _text, accepted) => { accepted(); return Promise.resolve("succeeded"); } });
  h.render();
  for (let i = 0; i < count; i++) { h.click(h.button("친구")); if (i < count - 1) h.click(h.button("다음")); }
  if (count > 1) h.click(h.button("답변 보내기"));
  assert.equal(nodes(h.tree).some(n => ["nav", "fieldset", "input", "button"].includes(n.type)), false);
  assert.match(h.tree.props.children, /응답을 기다리고/);
  messages = [...messages, { role: "user", content: "답변" }, { role: "assistant", content: "계획" }];
  await Promise.resolve(); h.render(); assert.equal(h.tree, null);
  messages = [h.props.message]; h.render();
  assert.equal(h.button("친구").props["aria-pressed"], true);
}));

test("manual composer pending collapses latest wizard without pretending it was a structured answer", () => harness(1, h => {
  global.__planningHooks.useChat = () => ({ messages: [h.props.message], pending: "시간: 18:30? 자유 답변", stayHere() {} });
  h.render(); assert.equal(nodes(h.tree).some(n => ["input", "button", "fieldset"].includes(n.type)), false);
  const { ChatUserContent } = h.require("./planning.cjs");
  assert.equal(ChatUserContent({content:"시간: 18:30? 자유 답변", planning:h.props.message.planning}).props.text, "시간: 18:30? 자유 답변");
}));

test("history attachment fallback normalizes identity without repeating inline URLs or merging queries", () => harness(1, h => {
  const { ChatUserContent } = h.require("./planning.cjs");
  const content = "앞 https://BLOG.naver.com/post?id=42#section 뒤";
  const attachments = [
    { kind: "url", url: "https://blog.naver.com/post?id=42" },
    { kind: "url", url: "https://blog.naver.com/post?id=43" },
    { kind: "url", url: "https://blog.naver.com/post?id=43" },
    { kind: "url", url: "javascript:alert(1)" },
  ];
  const tree = ChatUserContent({ content, attachments });
  assert.equal(tree.props.text, content + "\nhttps://blog.naver.com/post?id=43");
  assert.equal(tree.props.attachments, attachments);
}));

test("structured answer formatting preserves semantic values and ordinary colon text", () => harness(1, h => {
  const { planningAnswers } = h.require("./helper.cjs");
  const { ChatUserContent } = h.require("./planning.cjs");
  const planning = { questions: [{ question: "구장?" }, { question: "동행?" }, { question: "시간?" }, { question: "이동?" }] };
  const text = "구장?: 잠실\n동행?: 혼자\n시간?: 경기 전후 모두\n이동?: 대중교통·숙박 없음";
  assert.deepEqual(planningAnswers(text, planning), ["잠실", "혼자", "경기 전후 모두", "대중교통·숙박 없음"]);
  const tree = ChatUserContent({content:text, planning});
  assert.equal(nodes(tree).find(n=>n.type==="span").props.children, "잠실 · 혼자 · 경기 전후 모두 · 대중교통·숙박 없음");
  assert.equal(nodes(tree).find(n=>n.type==="details").props.open, undefined);
  assert.equal(nodes(tree).find(n=>n.type==="div").props.children.props.text, text);
  for (const ordinary of ["시간: 18:30? 숙박: 없음", "구장?: 잠실", "자유롭게 답해요"]) assert.equal(planningAnswers(ordinary, planning), undefined);
  assert.equal(planningAnswers(text), undefined);
  const multiline = {questions:[{question:"상세?"},{question:"이동?"}]};
  assert.deepEqual(planningAnswers("상세?: 18:30 도착\n경기 후 식사\n이동?: 버스: 2번", multiline), ["18:30 도착\n경기 후 식사", "버스: 2번"]);
}));

test("responsive ten-button choices precede one initially empty accessible input", () => harness(2, h => {
  h.props.message.planning.questions[0].choices = Array.from({ length: 10 }, (_, i) => `구장 ${i}`); h.render();
  const fieldset = nodes(h.tree).find(n => n.type === "fieldset");
  assert.equal(nodes(fieldset).filter(n => n.type === "button").length, 10);
  assert.ok(nodes(fieldset).filter(n => n.type === "button").every(n => n.props.type === "button" && typeof n.props["aria-pressed"] === "boolean"));
  assert.ok(nodes(fieldset).some(n => n.props?.className === "chat-question-choices"));
  const input = nodes(fieldset).find(n => n.type === "input");
  assert.equal(input.props.value, "");
  assert.equal(input.props["aria-label"], "질문 1 직접 입력");
  assert.equal(nodes(h.tree).filter(n => n.type === "input").length, 1);
  assert.equal(h.button("직접 입력하기"), undefined);
  const css = readFileSync(join(frontend, "styles/chat-planning.css"), "utf8");
  assert.match(css, /container-type:\s*inline-size/);
  for (const columns of [2, 3, 5]) assert.ok(css.includes(`repeat(${columns}, minmax(0, 1fr))`));
  assert.match(css, /@container \(min-width: 680px\)/);
  assert.match(css, /:focus-visible/);
  assert.match(css, /\.chat-question-choices button\s*\{[^}]*width:\s*100%[^}]*text-align:\s*left/);
  assert.match(css, /\.chat-planning button\s*\{[^}]*min-height:\s*44px/);
}));

test("choice then custom, clear, previous and next preserve independent drafts", () => harness(2, h => {
  const input = () => nodes(h.tree).find(n => n.type === "input");
  h.click(h.button("친구"));
  input().props.onFocus?.(); h.render();
  assert.equal(h.button("친구").props["aria-pressed"], true);
  input().props.onChange({ target: { value: "  직접 답변  " } }); h.render();
  assert.equal(h.button("친구").props["aria-pressed"], false);
  h.click(h.button("가족"));
  assert.equal(input().props.value, "  직접 답변  ");
  h.click(h.button("다음")); h.click(h.button("혼자")); h.click(h.button("이전"));
  assert.equal(h.button("가족").props["aria-pressed"], true);
  input().props.onFocus?.(); h.render();
  assert.equal(h.button("가족").props["aria-pressed"], true);
  input().props.onChange({ target: { value: "" } }); h.render();
  assert.equal(h.button("다음").props.disabled, true);
  input().props.onChange({ target: { value: "   " } }); h.render();
  assert.equal(h.button("다음").props.disabled, true);
  input().props.onChange({ target: { value: "최종 직접 답변" } }); h.render();
  h.click(h.button("다음"));
  assert.equal(h.button("혼자").props["aria-pressed"], true);
  assert.equal(h.sends.length, 0);
  h.click(h.button("답변 보내기"));
  assert.deepEqual(h.sends, [[2, "질문 1: 최종 직접 답변\n질문 2: 혼자"]]);
}));

test("LLM direct-input wording remains an ordinary immediate-send choice", () => harness(1, h => {
  h.props.message.planning.questions[0].choices.push("다른 구장 직접 입력"); h.render();
  h.click(h.button("다른 구장 직접 입력"));
  assert.deepEqual(h.sends, [[2, "질문 1: 다른 구장 직접 입력"]]);
}));

test("free-text question has no fabricated choices and sends without writer offer", () => harness(1, h => {
  h.props.message.planning = { offer_writer: false, questions: [{ question: "어느 날짜인가요?", choices: [] }] }; h.render();
  const { parsePlanning } = h.require("./helper.cjs");
  assert.deepEqual(parsePlanning(h.props.message.planning), h.props.message.planning);
  assert.equal(nodes(h.tree).filter(n => n.type === "button").length, 1);
  assert.equal(h.button("답변 보내기").props.disabled, true);
  assert.equal(nodes(h.tree).some(n => n.props?.children === "선택하면 바로 보내요"), false);
  nodes(h.tree).find(n => n.type === "input").props.onChange({ target: { value: "  10월 11일  " } }); h.render();
  h.click(h.button("답변 보내기"));
  assert.deepEqual(h.sends, [[2, "어느 날짜인가요?: 10월 11일"]]);
}));

test("single choice auto-sends once even with a stale double click", () => harness(1, h => {
  assert.equal(nodes(h.tree).some(n => n.type === "nav"), false);
  assert.equal(nodes(h.tree).find(n => n.type === "input").props.value, "");
  assert.equal(h.button("직접 입력하기"), undefined);
  assert.equal(h.button("답변 보내기"), undefined);
  const choice = h.button("친구");
  h.click(choice); h.click(choice);
  assert.deepEqual(h.sends, [[2, "질문 1: 친구"]]);
  assert.equal(h.tree.props["aria-busy"], true);
  assert.ok(h.stays > 0);
}));

test("single always-visible custom input requires explicit send; Enter cannot submit writer", () => harness(1, h => {
  assert.equal(h.button("답변 보내기"), undefined);
  let input = nodes(h.tree).find(n => n.type === "input");
  input.props.onFocus?.(); h.render();
  assert.equal(h.button("답변 보내기"), undefined);
  input.props.onChange({ target: { value: "   " } }); h.render();
  assert.equal(h.button("답변 보내기").props.disabled, true);
  input.props.onChange({ target: { value: "" } }); h.render();
  assert.equal(h.button("답변 보내기").props.disabled, true);
  input.props.onChange({ target: { value: "  기차  " } }); h.render();
  assert.equal(h.sends.length, 0);
  const oldInput = global.HTMLInputElement;
  global.HTMLInputElement = class {};
  try {
    let prevented = 0, stopped = 0;
    h.tree.props.onKeyDown({ key: "Enter", target: new global.HTMLInputElement(), nativeEvent: { isComposing: true }, preventDefault() { prevented++; }, stopPropagation() { stopped++; } });
    assert.deepEqual([prevented, stopped], [0, 1]);
  } finally { global.HTMLInputElement = oldInput; }
  const markup = h.require("react-dom/server").renderToStaticMarkup(h.require("react").createElement("form", null, h.tree));
  assert.equal((markup.match(/<form/g) ?? []).length, 1);
  assert.match(markup, /aria-label="질문 1 직접 입력"/);
  const send = h.button("답변 보내기");
  let prevented = 0, stopped = 0;
  const event = { preventDefault() { prevented++; }, stopPropagation() { stopped++; } };
  send.props.onClick(event); send.props.onClick(event);
  assert.deepEqual([prevented, stopped], [2, 2]);
  assert.deepEqual(h.sends, [[2, "질문 1: 기차"]]);
}));

for (const count of [2, 4]) test(`${count} steps keep answers editable and send only the final combined answers`, () => harness(count, h => {
  assert.equal(nodes(h.tree).filter(n => n.type === "fieldset").length, 1);
  assert.equal(h.button("이전").props.disabled, true);
  assert.equal(h.button("다음").props.disabled, true);
  h.click(h.button("친구"));
  assert.equal(h.button("친구").props["aria-pressed"], true);
  h.click(h.button("다음"));
  assert.equal(h.focuses, 1);
  assert.equal(nodes(h.tree).find(n => n.props?.["aria-current"] === "step").props.children, 2);
  assert.equal(nodes(h.tree).find(n => n.props?.["data-state"] === "completed").props.children, 1);
  nodes(h.tree).find(n => n.type === "input").props.onChange({ target: { value: "기차" } }); h.render();
  h.click(h.button("이전"));
  assert.equal(h.button("친구").props["aria-pressed"], true);
  h.click(h.button("가족")); h.click(h.button("다음"));
  assert.equal(nodes(h.tree).find(n => n.type === "input").props.value, "기차");
  for (let i = 2; i < count; i++) { h.click(h.button("다음")); h.click(h.button("혼자")); }
  assert.equal(h.sends.length, 0);
  const send = h.button("답변 보내기"); h.click(send); h.click(send);
  assert.deepEqual(h.sends, [[2, ["질문 1: 가족", "질문 2: 기차", ...Array.from({ length: count - 2 }, (_, i) => `질문 ${i + 3}: 혼자`)].join("\n")]]);
}));

test("direct draft survives choice switch; disabled/history cannot send; message key resets state", () => harness(2, h => {
  nodes(h.tree).find(n => n.type === "input").props.onChange({ target: { value: "오래된 입력" } }); h.render();
  h.click(h.button("친구"));
  assert.equal(h.button("친구").props["aria-pressed"], true);
  nodes(h.tree).find(n => n.type === "input").props.onFocus?.(); h.render();
  assert.equal(h.button("친구").props["aria-pressed"], true);
  assert.equal(nodes(h.tree).find(n => n.type === "input").props.value, "오래된 입력");
  h.props.disabled = true; h.render();
  assert.ok(nodes(h.tree).filter(n => n.type === "button").every(n => n.props.disabled));
  const stays = h.stays;
  h.click(h.button("다음")); assert.equal(h.stays, stays);
  h.props.disabled = false; h.props.message = { ...h.props.message, id: 99 }; h.render();
  assert.equal(nodes(h.tree).find(n => n.type === "input").props.value, "");
  assert.equal(h.button("다음").props.disabled, true);
  assert.equal(h.sends.length, 0);
}));

for (const outcome of ["rejected", "failed"]) test(`submission ${outcome} retains answers and enables change/retry`, () => harness(1, async h => {
  global.__planningHooks.useChat = () => ({ messages: [], pending: "", stayHere() {}, submitQuestions: async (_id, _text, accepted) => { if (outcome !== "rejected") accepted(); return outcome; } });
  h.render(); h.click(h.button("친구"));
  assert.equal(h.tree.props["aria-busy"], outcome === "failed");
  await Promise.resolve(); h.render();
  assert.equal(h.button("친구").props.disabled, false);
  assert.equal(h.button("친구").props["aria-pressed"], true);
  assert.ok(nodes(h.tree).some(n => n.props?.role === "status" && n.props.children.includes(outcome === "failed" ? "응답을 받지" : "보낼 수 없")));
  h.click(h.button("혼자")); await Promise.resolve(); h.render();
  assert.equal(h.button("혼자").props["aria-pressed"], true);
}));
for (const count of [1, 2, 4]) test(`Enter follows next/send for ${count} steps and ignores composition/repeat`, () => harness(count, h => {
  const previous = global.HTMLInputElement; global.HTMLInputElement = class {};
  try {
    const enter = (extra = {}) => h.tree.props.onKeyDown({key:"Enter", target:new global.HTMLInputElement(), nativeEvent:{}, preventDefault(){},stopPropagation(){}, ...extra});
    for (let i = 0; i < count; i++) {
      nodes(h.tree).find(n => n.type === "input").props.onChange({target:{value:`답변 ${i}`}}); h.render();
      enter({nativeEvent:{isComposing:true}}); enter({repeat:true}); h.render(); assert.equal(h.sends.length,0);
      enter(); h.render();
    }
    h.tree.props.onKeyDown?.({key:"Enter", target:new global.HTMLInputElement(), nativeEvent:{}, preventDefault(){},stopPropagation(){}}); assert.equal(h.sends.length,1);
  } finally {global.HTMLInputElement=previous;}
}));
