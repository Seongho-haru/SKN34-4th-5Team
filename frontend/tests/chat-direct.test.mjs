import assert from "node:assert/strict";
import { after, beforeEach, test } from "node:test";
import { createRequire } from "node:module";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const frontend = dirname(dirname(fileURLToPath(import.meta.url)));
const scratch = mkdtempSync(join(tmpdir(), "kbo-chat-direct-test-"));
after(() => rmSync(scratch, { recursive: true, force: true }));
for (const name of ["lib/course-directions", "lib/drawn-course", "lib/chat/current-course", "lib/member-auth-request", "lib/chat/planning", "lib/chat/types", "lib/chat/validation", "lib/chat/course", "lib/chat/history", "lib/chat/client"]) {
  const source = readFileSync(join(frontend, `${name}.ts`), "utf8");
  const { outputText } = ts.transpileModule(source, {
    fileName: `${name}.ts`, compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  });
  mkdirSync(dirname(join(scratch, `${name}.js`)), { recursive: true });
  writeFileSync(join(scratch, `${name}.js`), outputText);
}

global.window = { setTimeout, clearTimeout };
const stored = new Map();
global.sessionStorage = {
  getItem: key => stored.get(key) ?? null,
  setItem: (key, value) => stored.set(key, value),
  removeItem: key => stored.delete(key),
};
const require = createRequire(join(scratch, "entry.cjs"));
const { clearMemberTokens, saveMemberTokens } = require("./lib/member-auth-request.js");
const { fetchChatToolGroups, TOOL_GROUP_IDS, saveAnswerFeedback, fetchAdminFeedback, fetchAdminFeedbackDetail, ChatClientError, deleteChatMessages, deleteChatSession, editChatMessage, fetchChatHistory, getChatStatus, listChatSessions, renameChatSession, sendChatMessage } = require("./lib/chat/client.js");
const { restoreChatMessages } = require("./lib/chat/history.js");
const { courseToStops, parseChatCourse } = require("./lib/chat/course.js");
const { currentCourse } = require("./lib/chat/current-course.js");
const { parseChatRequest } = require("./lib/chat/validation.js");

test("tool menu accepts the server carry-in group without rejecting the entire menu", async () => {
  const groups = [{ id: "web_research", label: "웹 조사" }, { id: "carry_in", label: "반입 규정" }, { id: "day_plan", label: "코스" }];
  global.fetch = async () => json(groups);
  assert.deepEqual(await fetchChatToolGroups("guest"), groups);
  global.fetch = async () => json([...groups, { id: "unknown_group", label: "알 수 없음" }]);
  await assert.rejects(fetchChatToolGroups("guest"), error => error instanceof ChatClientError && error.status === 502);
});

test("writer metadata survives public response, history and follow-up including an explicitly cleared origin", () => {
  const writerState = { title: "내 코스 제목", origin: { lat: 37.5, lng: 127.1, name: "잠실새내역" }, completed: false };
  const place = { name: "카페", lat: 37.51, lng: 127.1, category: "CAFE", phase: "BEFORE", visitId: "cafe", label: "1" };
  const wire = { places: [place], stadiumCode: "JAMSIL", writerState, travel: { mode: "walk" } };
  const parsed = parseChatCourse(wire);
  assert.deepEqual(parsed.writerState, writerState);
  const current = { places: [place], stadiumCode: "JAMSIL", travelMode: "walk", legModes: {}, writerState: { ...writerState, title: "", origin: null } };
  const request = parseChatRequest({ messages: [{ role: "user", content: "카페만 바꿔줘" }], context: { currentCourse: current } });
  assert.deepEqual(request.context.currentCourse.writerState, current.writerState);
  const history = restoreChatMessages([{ id: 42, role: "assistant", content: "코스", status: "completed", tools: [], course: wire }]);
  assert.deepEqual(history[0].course.writerState, writerState);
  assert.throws(() => parseChatRequest({ messages: [{ role: "user", content: "변경" }], context: { currentCourse: { ...current, writerState: { ...writerState, completed: "yes" } } } }));
});
const { legKey } = require("./lib/course-directions.js");

test("completed visits and remaining schedule survive course card, map and follow-up requests", () => {
  const wire = { edit: true, stadiumCode: "JAMSIL", travel: { mode: "walk" }, legModes: {},
    game: { date: "2026-10-06", time: "18:30" }, progress: { startMinute: 1020, gameEndMinute: 1380 },
    places: [
      { visitId: "food", placeId: "1", name: "식당", category: "FOOD", phase: "BEFORE", lat: 37.51, lng: 127.08, time: "16:10", until: "17:00", completed: true },
      { visitId: "game", placeId: "2", name: "잠실", category: "STADIUM", phase: "GAME", lat: 37.512, lng: 127.071, time: "17:45", until: "23:00" },
    ] };
  const course = parseChatCourse(wire);
  const stops = courseToStops(course, () => "id");
  const snapshot = currentCourse(stops, "JAMSIL", "walk", {});
  const request = parseChatRequest({ messages: [{ role: "user", content: "출발 10분 늦어졌어" }], context: { currentCourse: snapshot } });
  assert.deepEqual(request.context.currentCourse.progress, wire.progress);
  assert.equal(request.context.currentCourse.places[0].completed, true);
  assert.equal(request.context.currentCourse.places[0].until, "17:00");
  const undone = courseToStops(parseChatCourse({ ...wire, progress: undefined, places: wire.places.map(p => ({ ...p, completed: undefined })) }), () => "id", stops);
  assert.equal(currentCourse(undone, "JAMSIL", "walk", {}).progress, undefined);
  assert.equal(undone[0].coursePlace.completed, undefined);
  for (const progress of [{ startMinute: -1 }, { gameEndMinute: 2880 }, { startMinute: true }, { arbitrary: 1 }]) {
    assert.throws(() => parseChatRequest({ messages: [{ role: "user", content: "변경" }], context: { currentCourse: { ...snapshot, progress } } }));
  }
});

test("the current map sends exact places, map numbers, game and per-leg modes after manual reorder", () => {
  const stops = [
    { name: "식당", category: "먹거리", visitId: "food", placeId: "10", lat: 37.511, lng: 127.075, isDrawnPoint: true },
    { name: "카페", category: "카페·디저트", visitId: "cafe", placeId: "20", lat: 37.513, lng: 127.078, isDrawnPoint: true },
    { name: "잠실", category: "야구장", visitId: "game", lat: 37.512, lng: 127.071, isDrawnPoint: true, courseGame: { date: "2026-10-06", time: "18:30" } },
  ];
  const order = [stops[1], stops[0], stops[2]];
  const modes = { [legKey(order[0], order[1])]: "transit", [legKey(stops[0], stops[1])]: "car" };
  const snapshot = currentCourse(order, "JAMSIL", "walk", modes);
  assert.deepEqual(snapshot.places.map(p => [p.visitId, p.label]), [["cafe", "출발"], ["food", "1"], ["game", "2"]]);
  assert.deepEqual(snapshot.legModes, { [legKey(order[0], order[1])]: "transit" });
  const request = parseChatRequest({ messages: [{ role: "user", content: "카페만 바꿔줘" }], context: { currentCourse: snapshot } });
  assert.deepEqual(request.context.currentCourse.game, stops[2].courseGame);
  assert.deepEqual(request.context.currentCourse.places.map(p => p.visitId), ["cafe", "food", "game"]);
  for (const bad of [{ ...snapshot, places: [snapshot.places[0], snapshot.places[0]] }, { ...snapshot, legModes: { nope: "car" } },
    { ...snapshot, places: [{ ...snapshot.places[0], lat: Infinity }] }]) {
    assert.throws(() => parseChatRequest({ messages: [{ role: "user", content: "바꿔" }], context: { currentCourse: bad } }));
  }
});

test("an edited course preserves fixed manual stops and changes only the requested venue", () => {
  const fixed = { name: "직접 고른 식당", category: "먹거리", lat: 37.51, lng: 127.08, placeId: "1", visitId: "food", tourContentId: "manual-reference" };
  const oldCafe = { name: "기존 카페", category: "카페·디저트", lat: 37.52, lng: 127.08, placeId: "2", visitId: "cafe", isDrawnPoint: true };
  const reply = parseChatCourse({ edit: true, stadiumCode: "JAMSIL", legModes: {}, travel: { mode: "walk" }, places: [
    { ...fixed, category: "FOOD", phase: "BEFORE", time: "15:30" },
    { ...oldCafe, name: "새 카페", lat: 37.515, placeId: "3", category: "CAFE", phase: "BEFORE", time: "16:30" },
  ] });
  const applied = courseToStops(reply, () => "unexpected", [fixed, oldCafe]);
  const { coursePlace, ...retained } = applied[0];
  assert.deepEqual(retained, fixed);
  assert.equal(coursePlace.time, "15:30");
  assert.equal(applied[1].visitId, "cafe");
  assert.equal(applied[1].placeId, "3");
  assert.equal(applied[1].name, "새 카페");
});

test("GPS or map origin remains separate from editable visits", () => {
  const origin = { name: "출발지", category: "출발", lat: 37.5, lng: 127.07, placeId: "route:origin", isDrawnPoint: true };
  const stop = { name: "카페", category: "카페·디저트", lat: 37.51, lng: 127.07, visitId: "cafe" };
  const snapshot = currentCourse([origin, stop], "JAMSIL", "walk", {});
  assert.equal(snapshot.places.length, 1);
  assert.equal(snapshot.places[0].label, "1");
  assert.equal(currentCourse([{ ...stop, category: "야구장" }], "JAMSIL", "walk", {}).places[0].category, "STADIUM");
});

test("origin-only and explicitly empty maps send authoritative writer context, never an empty recommendation", () => {
  const origin = { name: "출발지", category: "직접 지정", lat: 35.2, lng: 129.05, isMapPoint: true };
  for (const stops of [[origin], []]) {
    const writerState = { title: "", origin: stops.length ? { lat: origin.lat, lng: origin.lng, name: origin.name } : null, completed: false };
    const snapshot = currentCourse(stops, "SAJIK", "walk", {}, undefined, writerState);
    const request = parseChatRequest({ messages: [{ role: "user", content: "초밥 먹고 산책하다 구장 갈 코스 짜줘" }], context: { currentCourse: snapshot } });
    assert.deepEqual(request.context.currentCourse.places, []);
    assert.deepEqual(request.context.currentCourse.writerState, writerState);
    assert.equal(parseChatCourse({ ...snapshot, edit: true }), undefined);
    assert.throws(() => parseChatRequest({ messages: [{ role: "user", content: "코스" }], context: { currentCourse: { ...snapshot, writerState: undefined } } }));
  }
});

test("course source URLs are validated without inventing links for missing sources", () => {
  const place = { name: "카페", lat: 37.5, lng: 127, phase: "BEFORE", category: "CAFE" };
  for (const placeUrl of ["javascript:alert(1)", "data:text/html,bad", "//example.org", "https://secret@example.org/1", "https://example.org/\n1", undefined]) {
    assert.equal(parseChatCourse({ places: [{ ...place, placeUrl }] }).places[0].placeUrl, undefined);
  }
  assert.equal(parseChatCourse({ places: [{ ...place, placeUrl: "http://place.map.kakao.com/1" }] }).places[0].placeUrl, "https://place.map.kakao.com/1");
});
const json = (value, status = 200) => Response.json(value, { status });
const sse = events => new Response(new ReadableStream({
  start(controller) {
    controller.enqueue(new TextEncoder().encode(events.map(([event, data]) =>
      `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join("")));
    controller.close();
  },
}), { headers: { "Content-Type": "text/event-stream; charset=utf-8" } });
// Session ids and message ids are server-issued UUID strings (LangChain BaseMessage.id).
const SESSION = "3f2c1a4e-8b7d-4c21-9e0f-5a6b7c8d9e01";
const OTHER_SESSION = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d";
const USER_MSG = 1;
const ASSISTANT_MSG = 2;
const OTHER_MSG = 3;
const room = (id = SESSION, title = "첫 질문") => ({ id, title, created_at: "2026-09-28T00:00:00Z", updated_at: "2026-09-28T00:00:00Z" });
const row = (id, role, content, status = "completed", tools = []) => ({ id, sequence_no: id, role, content, status, tools, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" });
const tool = (id, tool_name, status) => ({ id, tool_name, status });
const answerEvents = (chunks = ["첫 ", "답변"], messageId = ASSISTANT_MSG, tools = []) => [
  ...chunks.map(text => ["delta", { text }]),
  ["done", { message_id: String(messageId), assistant_message: chunks.join(""), tools }],
];
const record = calls => async (url, init = {}) => {
  const call = { url: String(url), method: init.method ?? "GET", body: init.body ? JSON.parse(init.body) : undefined, headers: new Headers(init.headers), credentials: init.credentials };
  calls.push(call);
  return call;
};
const hanging = init => new Response(new ReadableStream({
  start(stream) {
    stream.enqueue(new TextEncoder().encode('event: delta\ndata: {"text": "부분 답"}\n\n'));
    init.signal.addEventListener("abort", () => stream.error(new DOMException("Aborted", "AbortError")), { once: true });
  },
}), { headers: { "Content-Type": "text/event-stream" } });

beforeEach(() => { stored.clear(); clearMemberTokens(); });

const ATTACHMENT = "11111111-1111-4111-8111-111111111111";
const attachmentDto = { id: ATTACHMENT, kind: "image", name: "seat.png", content_type: "image/png", size: 12, width: 1, height: 1, url: `/api/v2/chat/sessions/${SESSION}/attachments/${ATTACHMENT}/`, created_at: "2026-10-06T00:00:00Z" };
test("empty text MIME uploads use verified extension MIME without overriding multipart boundary", async () => {
  const { uploadChatAttachment, validateChatFile } = require("./lib/chat/client.js");
  for (const [name, type, expected] of [["notes.md", "", "text/markdown"], ["notes.txt", "", "text/plain"], ["notes.md", "text/x-markdown", "text/x-markdown"]]) {
    let captured;
    global.fetch = async (_, init) => { captured = init; return json({ ...attachmentDto, kind: "text", name, content_type: expected, width: null, height: null }, 201); };
    const file = new File(["notes"], name, { type });
    assert.equal(validateChatFile(file), "text"); await uploadChatAttachment("guest", SESSION, file);
    const uploaded = captured.body.get("file"); assert.equal(uploaded.name, name); assert.equal(uploaded.type, expected); assert.equal(await uploaded.text(), "notes");
    assert.equal(new Headers(captured.headers).get("Content-Type"), null);
  }
});
test("attachment multipart and private previews reuse Bearer; URL uploads are explicit JSON", async () => {
  const { uploadChatAttachment, fetchChatAttachmentBlob } = require("./lib/chat/client.js");
  saveMemberTokens("access-token", "refresh-token");
  const calls = [];
  global.fetch = async (url, init) => { calls.push({ url, init }); return init.method === "GET" ? new Response("image bytes") : json(attachmentDto, 201); };
  const image = new File(["image"], "seat.png", { type: "image/png" });
  const saved = await uploadChatAttachment("member", SESSION, image);
  assert.equal(saved.contentType, "image/png"); assert.equal(saved.createdAt, attachmentDto.created_at);
  assert.ok(calls[0].init.body instanceof FormData); assert.equal(calls[0].init.body.get("file").name, "seat.png");
  assert.equal(new Headers(calls[0].init.headers).get("Content-Type"), null);
  assert.equal(new Headers(calls[0].init.headers).get("Authorization"), "Bearer access-token");
  assert.equal(await (await fetchChatAttachmentBlob("member", SESSION, ATTACHMENT)).text(), "image bytes");
  const count = calls.length;
  await uploadChatAttachment("guest", SESSION, "https://example.com/info");
  assert.equal(calls.length, count + 1);
  assert.deepEqual(JSON.parse(calls.at(-1).init.body), { url: "https://example.com/info" });
  global.fetch = async () => json({ detail: "저장소를 사용할 수 없어요." }, 503);
  await assert.rejects(uploadChatAttachment("guest", SESSION, image), error => error.status === 503);
  await assert.rejects(uploadChatAttachment("guest", SESSION, new File(["svg"], "bad.svg", { type: "image/svg+xml" })), error => error.status === 400);
});
test("tool menu accepts the backend registry including carry_in and rejects invalid groups", async () => {
  const server = readFileSync(join(frontend, "../backend/llm/views/attachments.py"), "utf8");
  const groups = [...server.split("TOOL_GROUP_LABELS = {")[1].split("}")[0].matchAll(/"([a-z_]+)": "([^"]+)"/g)].map(([, id, label]) => ({ id, label }));
  assert.ok(groups.some(group => group.id === "carry_in" && group.label === "반입 규정"));
  global.fetch = async () => json(groups);
  const loaded = await fetchChatToolGroups("guest");
  assert.deepEqual(loaded, groups);
  assert.deepEqual(TOOL_GROUP_IDS, groups.map(group => group.id));
  for (const invalid of [[...groups, { id: "unknown", label: "Unknown" }], [{ id: 1, label: "Invalid" }], [{ id: "carry_in", label: null }], {}]) {
    global.fetch = async () => json(invalid);
    await assert.rejects(fetchChatToolGroups("guest"), error => error.status === 502);
  }
});

test("carry_in selection survives POST, PUT and history with strict unknown and duplicate validation", async () => {
  const calls = [], log = record(calls);
  global.fetch = async (url, init) => { await log(url, init); return sse(answerEvents()); };
  const body = { sessionId: SESSION, content: "반입 질문", toolGroupIds: ["carry_in"] };
  await sendChatMessage("guest", body);
  await editChatMessage("guest", { ...body, messageId: USER_MSG });
  assert.deepEqual(calls.map(call => call.body.tool_group_ids), [["carry_in"], ["carry_in"]]);
  assert.deepEqual(calls.map(call => call.method), ["POST", "PUT"]);
  for (const toolGroupIds of [["unknown"], ["carry_in", "carry_in"], [1], "carry_in"]) {
    await assert.rejects(sendChatMessage("guest", { ...body, toolGroupIds }), error => error.status === 400);
    await assert.rejects(editChatMessage("guest", { ...body, messageId: USER_MSG, toolGroupIds }), error => error.status === 400);
    global.fetch = async () => json([{ ...row(USER_MSG, "user", body.content), tool_group_ids: toolGroupIds }]);
    await assert.rejects(fetchChatHistory("guest", SESSION), error => error.status === 502);
  }
  assert.equal(calls.length, 2);
  global.fetch = async () => json([{ ...row(USER_MSG, "user", body.content), tool_group_ids: body.toolGroupIds }]);
  assert.deepEqual(restoreChatMessages(await fetchChatHistory("guest", SESSION))[0].toolGroupIds, ["carry_in"]);
});

test("message options map snake_case, PUT omissions preserve and explicit empty clears", async () => {
  const calls = [], log = record(calls); global.fetch = async (url, init) => { await log(url, init); return sse(answerEvents()); };
  await sendChatMessage("guest", { sessionId: SESSION, content: "question", toolGroupIds: ["rules", "weather", "carry_in"], attachmentIds: [ATTACHMENT] });
  assert.deepEqual(calls[0].body, { content: "question", tool_group_ids: ["rules", "weather", "carry_in"], attachment_ids: [ATTACHMENT] });
  await editChatMessage("guest", { sessionId: SESSION, messageId: 1, content: "edit" });
  assert.equal(Object.hasOwn(calls[1].body, "attachment_ids"), false);
  await editChatMessage("guest", { sessionId: SESSION, messageId: 1, content: "clear", toolGroupIds: [], attachmentIds: [] });
  assert.deepEqual(calls[2].body.attachment_ids, []);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "question", toolGroupIds: ["unknown"] }), error => error.status === 400);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "question", attachmentIds: [ATTACHMENT, ATTACHMENT] }), error => error.status === 400);
});
for (const mode of ["guest", "member"]) for (const method of ["POST", "PUT"]) for (const count of [9, 10, 11]) {
  test(`${mode} ${method} accepts 9/10 attachments and rejects 11: ${count}`, async () => {
    if (mode === "member") saveMemberTokens("access-token", "refresh-token");
    const calls = [], log = record(calls);
    global.fetch = async (url, init) => { await log(url, init); return sse(answerEvents()); };
    const attachmentIds = Array.from({ length: count }, (_, index) => `11111111-1111-4111-8111-${String(index + 1).padStart(12, "0")}`);
    const body = { sessionId: SESSION, content: "question", attachmentIds };
    const send = () => method === "POST" ? sendChatMessage(mode, body) : editChatMessage(mode, { ...body, messageId: USER_MSG });
    if (count === 11) {
      await assert.rejects(send(), error => error instanceof ChatClientError && error.status === 400 && !error.uncertain);
      assert.equal(calls.length, 0);
    } else {
      assert.equal((await send()).reply, "첫 답변");
      assert.equal(calls.length, 1);
      assert.equal(calls[0].method, method);
      assert.equal(calls[0].url, `/api/v2/chat/sessions/${SESSION}/messages/`);
      assert.deepEqual(calls[0].body, { ...(method === "PUT" ? { message_id: USER_MSG } : {}), content: "question", attachment_ids: attachmentIds });
      assert.equal(calls[0].headers.get("Authorization"), mode === "member" ? "Bearer access-token" : null);
    }
  });
}

test("history restores validated metadata and manual groups, rejects unsafe private preview URL", async () => {
  global.fetch = async () => json([{ ...row(1, "user", "image"), attachments: [attachmentDto], tool_group_ids: ["stadium_info"] }]);
  const message = restoreChatMessages(await fetchChatHistory("guest", SESSION))[0];
  assert.equal(message.attachments[0].contentType, "image/png"); assert.deepEqual(message.toolGroupIds, ["stadium_info"]);
  global.fetch = async () => json([{ ...row(1, "user", "image"), attachments: [{ ...attachmentDto, url: "https://evil.test/image" }] }]);
  await assert.rejects(fetchChatHistory("guest", SESSION), error => error.status === 502);
});

test("SSE keep-alive comments are ignored between split answer frames and never become chat text", async () => {
  const deltas = [];
  const body = ': keep-alive\n\n: connection alive\r\n\r\n: comment\nevent: delta\ndata: {"text":"완성"}\n\n: keep-alive\n\nevent: done\ndata: {"message_id":"2","assistant_message":"완성","tools":[]}\n\n: keep-alive\n\n';
  const bytes = new TextEncoder().encode(body);
  global.fetch = async () => new Response(new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7));
    controller.close();
  } }), { headers: { "Content-Type": "text/event-stream" } });
  const reply = await sendChatMessage("guest", { sessionId: SESSION, content: "코스" }, undefined, { onDelta: piece => deltas.push(piece) });
  assert.equal(reply.reply, "완성"); assert.deepEqual(deltas, ["완성"]);
});

test("keep-alives alone cannot turn an incomplete response into success", async () => {
  global.fetch = async () => new Response(': keep-alive\n\n: keep-alive\n\n', { headers: { "Content-Type": "text/event-stream" } });
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "코스" }), error => error instanceof ChatClientError && error.uncertain);
});

test("v2 completed course and timing warning survive SSE and history and become map stops", async () => {
  const course = { stadiumCode: "JAMSIL", origin: { lat: 37.55, lng: 126.97, name: "서울역" }, entryPoint: { lat: 37.512, lng: 127.044 }, approachNotice: "서쪽 진입점부터 코스를 골랐어요.", timeWarning: "2번째 장소부터는 경기 전에 방문하기 어려워요.", places: [
    { name: "잠실 식당", category: "FOOD", phase: "BEFORE", lat: 37.511, lng: 127.075, placeId: "1", placeUrl: "https://place.map.kakao.com/1" },
    { name: "잠실야구장", category: "STADIUM", phase: "GAME", lat: 37.512, lng: 127.071 },
  ] };
  global.fetch = async () => sse([["done", { message_id: "2", assistant_message: "완성", tools: [], course }]]);
  const reply = await sendChatMessage("guest", { sessionId: SESSION, content: "코스", context: { stadium: "JAMSIL" } });
  assert.equal(reply.course.stadiumCode, "JAMSIL");
  assert.equal(reply.course.timeWarning, course.timeWarning);
  assert.deepEqual(reply.course.origin, course.origin);
  assert.deepEqual(reply.course.entryPoint, course.entryPoint);
  assert.equal(reply.course.approachNotice, course.approachNotice);
  assert.equal(reply.course.places[0].placeUrl, course.places[0].placeUrl);
  assert.equal(courseToStops(reply.course, () => "visit").length, 2);
  const previous = [{ id: 2, role: "assistant", content: "완성", course: reply.course }];
  const restored = restoreChatMessages([{ ...row(2, "assistant", "완성"), course }], previous);
  assert.equal(restored[0].course, reply.course); // 지도 담기/되돌리기 상태도 유지
  assert.equal(restoreChatMessages([{ ...row(2, "assistant", "완성"), course }])[0].course.timeWarning, course.timeWarning);
  assert.equal(restoreChatMessages([{ ...row(2, "assistant", "완성"), course }])[0].course.places[0].placeUrl, course.places[0].placeUrl);
  global.fetch = async () => sse([["done", { message_id: "2", assistant_message: "텍스트", tools: [], course: { places: [{ lat: "bad" }] } }]]);
  assert.equal((await sendChatMessage("guest", { sessionId: SESSION, content: "코스" })).course, undefined);
});

test("conversation conditions survive failed replacement, SSE and history without requiring a course", async () => {
  const coursePreferences = { conditions: ["카페는 스타벅스"], lockedPlaces: ["식당"], rejectedPlaces: ["이전 카페"] };
  global.fetch = async () => sse([["done", { message_id: "2", assistant_message: "조건 미확인, 코스 유지", tools: [], coursePreferences }]]);
  const reply = await sendChatMessage("guest", { sessionId: SESSION, content: "카페 바꿔줘" });
  assert.equal(reply.course, undefined);
  assert.deepEqual(reply.coursePreferences, coursePreferences);
  assert.deepEqual(restoreChatMessages([{ ...row(2, "assistant", reply.reply), coursePreferences }])[0].coursePreferences, coursePreferences);
  const empty = { conditions: [], lockedPlaces: [], rejectedPlaces: [] };
  assert.deepEqual(restoreChatMessages([{ ...row(2, "assistant", "해제"), coursePreferences: empty }])[0].coursePreferences, empty);
  assert.equal(restoreChatMessages([{ ...row(2, "assistant", "실패", "failed"), coursePreferences }])[0].coursePreferences, undefined);
});

test("custom stay duration and convenience category survive card, map and next request", () => {
  const course = parseChatCourse({ edit: true, stadiumCode: "JAMSIL", places: [
    { visitId: "shop", name: "편의점", lat: 37.512, lng: 127.07, category: "CONVENIENCE", phase: "BEFORE", stayOverride: 20 },
    { visitId: "game", name: "잠실", lat: 37.51, lng: 127.07, category: "STADIUM", phase: "GAME" },
  ] });
  const stops = courseToStops(course, () => "new");
  assert.equal(stops[0].category, "편의점");
  const context = currentCourse(stops, "JAMSIL", "walk", {});
  const request = parseChatRequest({ messages: [{ role: "user", content: "순서 바꿔줘" }], context: { currentCourse: context } });
  assert.equal(request.context.currentCourse.places[0].stayOverride, 20);
  const stadiumOnly = parseChatCourse({ ...course, places: [course.places[1]] });
  assert.equal(stadiumOnly.places.length, 1);
  assert.ok(parseChatRequest({ messages: [{ role: "user", content: "카페 추가" }], context: { currentCourse: currentCourse(courseToStops(stadiumOnly), "JAMSIL", "walk", {}) } }));
});

test("member send creates a UUID session then streams v2 delta/done with Bearer auth", async () => {
  saveMemberTokens("access-token", "refresh-token");
  const calls = [], log = record(calls);
  global.fetch = async (url, init = {}) => {
    const call = await log(url, init);
    if (call.method === "GET") return json([]);
    if (call.url === "/api/v2/chat/sessions/") return json(room(), 201);
    return sse(answerEvents());
  };
  assert.equal((await getChatStatus("member")).provider, "backend");
  const seen = [];
  const reply = await sendChatMessage("member", { content: "  첫 질문 ", context: { stadium: "잠실야구장", intent: "route", origin: { lat: 37.5, lng: 127.07 } } }, undefined, { onDelta: value => seen.push(value) });
  assert.deepEqual(seen, ["첫 ", "답변"]);
  assert.deepEqual({ reply: reply.reply, sessionId: reply.sessionId, assistant: reply.assistantMessageId, provider: reply.provider },
    { reply: "첫 답변", sessionId: SESSION, assistant: ASSISTANT_MSG, provider: "backend" });
  assert.deepEqual(calls.map(call => [call.method, call.url]), [
    ["GET", "/api/v2/chat/sessions/"], ["POST", "/api/v2/chat/sessions/"], ["POST", `/api/v2/chat/sessions/${SESSION}/messages/`],
  ]);
  assert.deepEqual(calls[1].body, { title: "첫 질문" });
  // Only the new question is sent: the server owns history. Context goes as the structured object v2 reads.
  assert.deepEqual(calls[2].body, { content: "첫 질문", context: { stadium: "잠실야구장", intent: "route", origin: { lat: 37.5, lng: 127.07 } } });
  assert.equal(calls[2].headers.get("Accept"), "text/event-stream");
  assert.ok(calls.every(call => call.headers.get("Authorization") === "Bearer access-token"));
});

test("guest session is cookie-owned: no Authorization, no credentials override, and reload lists the same UUID", async () => {
  const calls = [], log = record(calls);
  let cookieIssued = false;
  global.fetch = async (url, init = {}) => {
    const call = await log(url, init);
    if (call.url === "/api/v2/chat/sessions/" && call.method === "POST") { cookieIssued = true; return json(room(SESSION, "비회원 질문"), 201); }
    if (call.url === "/api/v2/chat/sessions/") return json(cookieIssued ? [room(SESSION, "비회원 질문")] : []);
    if (call.method === "GET") return json([row(USER_MSG, "user", "비회원 질문"), row(ASSISTANT_MSG, "assistant", "비회원 답")]);
    return sse(answerEvents(["비회원 ", "답"], ASSISTANT_MSG));
  };
  assert.deepEqual(await listChatSessions("guest"), []);
  const reply = await sendChatMessage("guest", { content: "비회원 질문" });
  assert.equal(reply.provider, "guest");
  assert.equal(reply.sessionId, SESSION);
  // Reload: the browser resends the HttpOnly guest_id cookie, so the list and history come back.
  const sessions = await listChatSessions("guest");
  assert.deepEqual(sessions.map(item => item.id), [SESSION]);
  const restored = restoreChatMessages(await fetchChatHistory("guest", sessions[0].id));
  assert.deepEqual(restored, [
    { id: USER_MSG, role: "user", content: "비회원 질문", status: "completed" },
    { id: ASSISTANT_MSG, role: "assistant", content: "비회원 답", status: "completed" },
  ]);
  assert.ok(calls.every(call => call.headers.get("Authorization") === null && call.credentials === undefined));
  assert.ok(calls.every(call => call.url.startsWith("/api/v2/chat/sessions/")));
});

test("edit PUTs message_id and content, then streams the regenerated answer", async () => {
  saveMemberTokens("access-token", "refresh-token");
  const calls = [], log = record(calls);
  global.fetch = async (url, init = {}) => { await log(url, init); return sse(answerEvents(["고친 ", "답"], OTHER_MSG)); };
  const reply = await editChatMessage("member", { sessionId: SESSION, messageId: USER_MSG, content: "고친 질문", context: { intent: "baseball" } });
  assert.deepEqual([calls[0].method, calls[0].url], ["PUT", `/api/v2/chat/sessions/${SESSION}/messages/`]);
  assert.deepEqual(calls[0].body, { message_id: USER_MSG, content: "고친 질문", context: { intent: "baseball" } });
  assert.deepEqual({ reply: reply.reply, id: reply.assistantMessageId }, { reply: "고친 답", id: OTHER_MSG });
  await assert.rejects(editChatMessage("member", { sessionId: SESSION, messageId: "not-a-uuid", content: "x" }), error => error instanceof ChatClientError && error.status === 400);
});

test("delete truncates with a DELETE body and accepts the empty 204", async () => {
  const calls = [], log = record(calls);
  global.fetch = async (url, init = {}) => {
    const call = await log(url, init);
    if (call.body?.message_id === OTHER_MSG) return json({ detail: "No ChatMessage matches the given query." }, 404);
    return new Response(null, { status: 204 });
  };
  assert.equal(await deleteChatMessages("guest", SESSION, USER_MSG), undefined);
  assert.deepEqual([calls[0].method, calls[0].url, calls[0].body], ["DELETE", `/api/v2/chat/sessions/${SESSION}/messages/`, { message_id: USER_MSG }]);
  assert.equal(calls[0].headers.get("Content-Type"), "application/json");
  await assert.rejects(deleteChatMessages("guest", SESSION, OTHER_MSG), error => error instanceof ChatClientError && error.status === 404 && !error.uncertain);
});

test("session rename/delete/history use UUID paths, stopped status and tool calls in the current DTOs", async () => {
  saveMemberTokens("access-token", "refresh-token");
  const calls = [], log = record(calls);
  global.fetch = async (url, init = {}) => {
    const call = await log(url, init);
    if (call.method === "DELETE") return new Response(null, { status: 204 });
    if (call.method === "PATCH") return json(room(SESSION, call.body.title));
    return json([
      row(USER_MSG, "user", "질문", "stopped"),
      row(ASSISTANT_MSG, "assistant", "답", "completed", [tool("call-1", "search_places", "completed")]),
    ]);
  };
  assert.equal((await renameChatSession("member", SESSION, "이름")).title, "이름");
  const history = await fetchChatHistory("member", SESSION);
  assert.deepEqual(restoreChatMessages(history).map(item => [item.id, item.role, item.status]), [[USER_MSG, "user", "stopped"], [ASSISTANT_MSG, "assistant", "completed"]]);
  assert.deepEqual(restoreChatMessages(history)[1].tools, [{ id: "call-1", toolName: "search_places", status: "completed", kind: "tool", parentId: null }]);
  assert.equal(await deleteChatSession("member", SESSION), undefined);
  assert.deepEqual(calls.map(call => [call.method, call.url]), [
    ["PATCH", `/api/v2/chat/sessions/${SESSION}/`], ["GET", `/api/v2/chat/sessions/${SESSION}/messages/`], ["DELETE", `/api/v2/chat/sessions/${SESSION}/`],
  ]);
  assert.ok(calls.every(call => call.headers.get("Authorization") === "Bearer access-token"));
  await assert.rejects(fetchChatHistory("member", "7"), error => error instanceof ChatClientError && error.status === 400);
  global.fetch = async () => json([{ id: "not-a-uuid-but-string-is-fine", title: 5 }]);
  await assert.rejects(listChatSessions("member"), error => error instanceof ChatClientError && error.status === 502);
  global.fetch = async () => json([{ ...row(USER_MSG, "human", "옛 역할") }]);
  await assert.rejects(fetchChatHistory("member", SESSION), error => error instanceof ChatClientError && error.status === 502);
  global.fetch = async () => json([{ ...row(USER_MSG, "user", "내부 상태", "cancelled") }]);
  await assert.rejects(fetchChatHistory("member", SESSION), error => error instanceof ChatClientError && error.status === 502);
  global.fetch = async () => json([{ ...row("42", "user", "숫자 ID") }]);
  await assert.rejects(fetchChatHistory("member", SESSION), error => error instanceof ChatClientError && error.status === 502);
  global.fetch = async () => json([{ ...row("not-a-uuid", "user", "잘못된 ID") }]);
  await assert.rejects(fetchChatHistory("member", SESSION), error => error instanceof ChatClientError && error.status === 502);
});

test("tool SSE events are accepted and reach onTool with id/tool_name/status; done carries the final tools array", async () => {
  const seenTools = [];
  global.fetch = async () => sse([
    ["tool", tool("call-1", "search_places", "running")],
    ["delta", { text: "답변 " }],
    ["tool", tool("call-1", "search_places", "completed")],
    ["delta", { text: "완성" }],
    ["done", { message_id: String(ASSISTANT_MSG), assistant_message: "답변 완성", tools: [tool("call-1", "search_places", "completed")] }],
  ]);
  const reply = await sendChatMessage("guest", { sessionId: SESSION, content: "질문" }, undefined, { onTool: value => seenTools.push(value) });
  assert.deepEqual(seenTools, [tool("call-1", "search_places", "running"), tool("call-1", "search_places", "completed")]);
  assert.deepEqual(reply.tools, [{ id: "call-1", toolName: "search_places", status: "completed", kind: "tool", parentId: null }]);
});

test("a done frame with a non-integer or empty message_id is rejected as uncertain", async () => {
  global.fetch = async () => sse([["delta", { text: "답" }], ["done", { message_id: "22222222-2222-4222-8222-222222222222", assistant_message: "답", tools: [] }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && error.uncertain);
  global.fetch = async () => sse([["delta", { text: "답" }], ["done", { message_id: "", assistant_message: "답", tools: [] }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && error.uncertain);
});

test("a tool frame with an unknown status or missing id/tool_name is rejected as uncertain", async () => {
  global.fetch = async () => sse([["tool", { id: "call-1", tool_name: "search_places", status: "unknown" }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && error.uncertain);
  global.fetch = async () => sse([["tool", { id: "", tool_name: "search_places", status: "running" }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && error.uncertain);
});

test("stream error frame is uncertain: the final save may or may not have committed", async () => {
  global.fetch = async () => sse([["delta", { text: "부분" }], ["error", { detail: "답변 생성에 실패했습니다. 다시 시도해 주세요." }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error =>
    error instanceof ChatClientError && error.message === "답변 생성에 실패했습니다. 다시 시도해 주세요." && error.uncertain && error.sessionId === SESSION);
});

test("stream that closes without done or error (superseded by edit/delete) is reported, never shown as saved", async () => {
  global.fetch = async () => sse([["delta", { text: "지워진 턴" }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error =>
    error instanceof ChatClientError && error.uncertain && /연결이 끊겼/.test(error.message));
  global.fetch = async () => sse([["delta", { text: "답" }], ["done", { message_id: String(ASSISTANT_MSG), assistant_message: "답" }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && /최종 답변/.test(error.message));
  global.fetch = async () => sse([["checkpoint", { turn_id: "old", receipt: "x" }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && /알 수 없는/.test(error.message));
});

test("SSE reader handles split frames and UTF-8 boundaries", async () => {
  const bytes = new TextEncoder().encode(answerEvents(["잠실 ", "야구장"], OTHER_MSG).map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join(""));
  global.fetch = async () => new Response(new ReadableStream({
    start(stream) { for (let index = 0; index < bytes.length; index += 5) stream.enqueue(bytes.slice(index, index + 5)); stream.close(); },
  }), { headers: { "Content-Type": "text/event-stream" } });
  const reply = await sendChatMessage("guest", { sessionId: SESSION, content: "질문" });
  assert.deepEqual([reply.reply, reply.assistantMessageId], ["잠실 야구장", OTHER_MSG]);
});

test("server stopped is a known terminal event with no partial reply", async () => {
  global.fetch = async () => sse([["delta", { text: "부분" }], ["stopped", {}]]);
  const seen = [];
  await assert.rejects(
    sendChatMessage("guest", { sessionId: SESSION, content: "질문" }, undefined, { onDelta: piece => seen.push(piece) }),
    error => error instanceof ChatClientError && error.status === 409 && !error.uncertain && error.sessionId === SESSION,
  );
  assert.deepEqual(seen, ["부분"]);
});

test("pre-stream JSON errors surface the DRF detail", async () => {
  global.fetch = async () => json({ content: ["이 필드는 2200자 이하여야 합니다."] }, 400);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && error.status === 400 && /2200/.test(error.message));
  global.fetch = async () => json({ detail: "session not found" }, 404);
  await assert.rejects(sendChatMessage("guest", { sessionId: OTHER_SESSION, content: "질문" }), error => error instanceof ChatClientError && error.status === 404);
});

test("Stop is a local abort: it rejects with 499 and returns no saved partial answer", async () => {
  const controller = new AbortController();
  let received = "";
  global.fetch = async (url, init = {}) => hanging(init);
  await assert.rejects(
    sendChatMessage("guest", { sessionId: SESSION, content: "질문" }, controller.signal, { onDelta: answer => { received = answer; controller.abort(); } }),
    error => error instanceof ChatClientError && error.status === 499,
  );
  assert.equal(received, "부분 답");
});

const coursePlaces = [
  { phase: "BEFORE", name: "상무초밥 잠실점", lat: 37.51, lng: 127.08, category: "FOOD", placeId: "123456", address: "서울 송파구", reason: "초밥", time: "16:06", stayMin: 50 },
  { phase: "BEFORE", name: "좌표 없는 곳", lat: null, lng: 127.08, category: "CAFE" },
  { phase: "GAME", name: "잠실야구장", lat: 37.512, lng: 127.072, category: "STADIUM", placeId: null, time: "17:45" },
  { phase: "AFTER", name: "잠실 게스트하우스", lat: 37.51, lng: 127.08, category: "STAY", placeId: "555", time: "22:49" },
];
test("course parsing ignores non-course answers and builds editable stops", () => {
  assert.equal(parseChatCourse({ assistant_message: "LG는 3위예요" }), undefined);
  assert.equal(parseChatCourse({ places: [{ phase: "GAME", name: "잠실야구장", lat: 37.5, lng: 127, category: "STADIUM" }] }), undefined);
  const course = parseChatCourse({ places: coursePlaces, stadiumCode: "JAMSIL", travel: { mode: "walk" } });
  assert.equal(course.travelMode, "walk");
  let id = 0;
  const stops = courseToStops(course, () => `id-${++id}`);
  assert.deepEqual(stops.map(stop => [stop.name, stop.category, stop.placeId, stop.isDrawnPoint]), [
    ["상무초밥 잠실점", "먹거리", "123456", true],
    ["잠실야구장", "야구장", "chat:stadium:JAMSIL", true],
    ["잠실 게스트하우스", "숙박", "555", true],
  ]);
  assert.ok(stops.every(stop => stop.visitId));
});

test("collected place identities survive chat parsing and route pin conversion", () => {
  const id = "collected:SBIZ:" + "a".repeat(80);
  const course = parseChatCourse({ places: [{ ...coursePlaces[0], placeId: id }, coursePlaces[2]], stadiumCode: "JAMSIL" });
  const stops = courseToStops(course, () => "visit-1");
  assert.equal(stops[0].placeId, id);
  assert.equal(stops[0].isDrawnPoint, true);
  assert.equal(stops[1].category, "야구장");
});

test("member auth failure never falls back to the guest cookie", async () => {
  global.fetch = async () => { throw new Error("no request may be sent without a member token"); };
  await assert.rejects(getChatStatus("member"), error => error instanceof ChatClientError && error.status === 401);
  await assert.rejects(sendChatMessage("member", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && error.status === 401);
});

test("provider enables guest chat, keeps stop local and clears state on every identity switch", () => {
  const provider = readFileSync(join(frontend, "components/chat-provider.tsx"), "utf8");
  const surfaces = ["components/chat-popup.tsx", "components/chat-workspace.tsx"].map(path => readFileSync(join(frontend, path), "utf8"));
  assert.match(provider, /const identity = memberStatus === "authenticated"/);
  assert.match(provider, /memberStatus === "authenticated" \? "member" : memberStatus === "anonymous" \? "guest" : null/);
  for (const cleanup of ["controller.abort()", "backendSessions.current.clear()", "archivedConversations.current.clear()", "historyRef.current = []", "syncRequestRef.current?.abort()"]) assert.ok(provider.includes(cleanup));
  assert.match(provider, /받던 답변은 저장되지 않아요/);
  assert.match(provider, /이 질문 이후의 대화는 모두 지워지고/);
  assert.match(provider, /이 질문과 이후 대화를 모두 지울까요/);
  assert.doesNotMatch(provider, /localStorage|sessionStorage|document\.cookie|credentials/);
  assert.doesNotMatch(provider, /checkpoint|finalize|fetchChatTurns|sendGuestChatMessage|받은 답변까지/);
  assert.match(provider, /messages: identityChanged \? \[\] : messages/);
  for (const surface of surfaces) {
    assert.match(surface, /답변 생성 중단/);
    assert.match(surface, /받던 답변은 저장되지 않아요/);
    assert.match(surface, /aria-relevant="additions"/);
    assert.match(surface, /message\.id !== undefined && available && !busy/);
    assert.doesNotMatch(surface, /로그인하고 질문하기|조회만 할 수 있어요|받은 답변까지 보관/);
  }
});

test("authenticated chat has no legacy Next cookie relay", () => {
  assert.equal(existsSync(join(frontend, "app/chat-api/route.ts")), false);
  assert.equal(existsSync(join(frontend, "lib/chat/team.ts")), false);
  assert.equal(existsSync(join(frontend, "app/baseball-admin-api/route.ts")), false);
});

test("client has no retired v1 guest, turns, finalize or non-stream endpoints", () => {
  const client = readFileSync(join(frontend, "lib/chat/client.ts"), "utf8");
  assert.doesNotMatch(client, /\/api\/v1\/|guest\/|turns|finalize|checkpoint|contextPrefix|credentials|document\.cookie/);
});

test("LLM streams skip the 55s total timer while plain API calls keep it", async () => {
  const delays = [];
  const original = global.window.setTimeout;
  global.window.setTimeout = (fn, ms) => { delays.push(ms); return original(fn, ms); };
  try {
    global.fetch = async (url, init = {}) => (init.method === "GET" ? json([]) : sse(answerEvents()));
    await sendChatMessage("guest", { sessionId: SESSION, content: "질문" });
    await editChatMessage("guest", { sessionId: SESSION, messageId: USER_MSG, content: "수정" });
    assert.deepEqual(delays, []);
    await listChatSessions("guest");
    assert.deepEqual(delays, [55_000]);
  } finally {
    global.window.setTimeout = original;
  }
});

test("appendTimeline coalesces deltas, keeps delta→tool→delta order, and updates tool ids in place", () => {
  const { appendTimeline } = require("./lib/chat/types.js");
  const tool = status => ({ id: "t1", toolName: "search_places", status });
  let items = [];
  for (const event of ["가", "나", tool("running"), { id: "t2", toolName: "get_weather", status: "running" }, "", tool("completed"), "다"]) items = appendTimeline(items, event);
  assert.deepEqual(items, [
    { kind: "text", text: "가나", parentId: null },
    { kind: "tools", tools: [tool("completed"), { id: "t2", toolName: "get_weather", status: "running" }] },
    { kind: "text", text: "다", parentId: null },
  ]);
});

test("done.assistant_message longer than MAX_REPLY_LENGTH is rejected even with no deltas", async () => {
  global.fetch = async () => sse([["done", { message_id: String(ASSISTANT_MSG), assistant_message: "가".repeat(8001), tools: [] }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error instanceof ChatClientError && error.uncertain);
});

test("usage DTO is fetched from the owner endpoint and quota errors keep a stable code", async () => {
  const { fetchChatUsage, USAGE_EXHAUSTED } = require("./lib/chat/client.js");
  const dto = { plan: "guest", period: "lifetime", timezone: null, resets_at: null, tokens_per_credit: 1000, limit_tokens: 10000,
    used_tokens: 1235, reserved_tokens: 0, remaining_tokens: 8765, remaining_credits: "8.765", can_send: true, active_turn: false, unknown_calls: 0, accounting_state: "known" };
  const calls = [], log = record(calls);
  global.fetch = async (url, init = {}) => { await log(url, init); return json(dto); };
  assert.deepEqual(await fetchChatUsage("guest"), dto);
  assert.deepEqual(calls.map(call => [call.method, call.url]), [["GET", "/api/v2/chat/usage/"]]);
  global.fetch = async () => json({ ...dto, used_tokens: -1 });
  await assert.rejects(fetchChatUsage("guest"), error => error instanceof ChatClientError && error.status === 502);

  global.fetch = async (url, init = {}) => (init.method ?? "GET") === "GET" ? json([room()]) : json({ code: USAGE_EXHAUSTED, detail: "사용 가능한 크레딧을 모두 사용했어요." }, 402);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error.status === 402 && error.code === USAGE_EXHAUSTED && error.message === "사용 가능한 크레딧을 모두 사용했어요.");
  const { USAGE_BUSY } = require("./lib/chat/client.js");
  global.fetch = async () => json({ code: USAGE_BUSY, detail: "진행 중인 답변이 끝난 뒤 다시 시도해 주세요." }, 409);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error.status === 409 && error.code === USAGE_BUSY && !error.uncertain && error.message === "진행 중인 답변이 끝난 뒤 다시 시도해 주세요.");
  global.fetch = async (url, init = {}) => (init.method ?? "GET") === "GET" ? json([room()]) : sse([["delta", { text: "부분" }], ["error", { detail: "사용 가능한 크레딧을 모두 사용했어요.", code: USAGE_EXHAUSTED }]]);
  await assert.rejects(sendChatMessage("guest", { sessionId: SESSION, content: "질문" }), error => error.code === USAGE_EXHAUSTED && error.uncertain);
});


test("planning payload is validated on SSE and history; invalid UI keeps text fallback", async () => {
  const payload = { offer_writer: true, questions: [{ question: "동행", choices: ["혼자", "친구"] }, { question: "이동", choices: ["도보", "차", "버스", "미정"] }] };
  const valid = [{ offer_writer: false, questions: [{ question: "구장", choices: Array.from({ length: 10 }, (_, i) => `구장 ${i}`) }] }, payload, { offer_writer: false, questions: [{ question: "어느 날짜인가요?", choices: [] }] }, { offer_writer: false, questions: [{ question: "어느 팀인가요?", choices: ["두산", "LG"] }] }];
  for (const planning of [...valid, { offer_writer: false, questions: [] }, { offer_writer: true, questions: [{ question: "bad", choices: ["one"] }] }, { offer_writer: true, questions: [{ question: "bad", choices: Array.from({ length: 11 }, (_, i) => `${i}`) }] }, { offer_writer: "yes", questions: [] }]) {
    const seen = [];
    const feedback = { rating: "up", reason: "", comment: "" };
    global.fetch = async (url, init) => init.method === "GET" ? json([{ ...row(2, "assistant", "텍스트"), planning, feedback }, { ...row(3, "user", "삭제된 답변의 질문"), answer_deleted: true }]) : sse([["planning", planning], ["done", { message_id: "2", assistant_message: "텍스트", tools: [], planning }]]);
    const reply = await sendChatMessage("guest", { sessionId: SESSION, content: "계획" }, undefined, { onPlanning: value => seen.push(value) });
    const history = restoreChatMessages(await fetchChatHistory("guest", SESSION));
    assert.equal(reply.reply, "텍스트");
    assert.deepEqual(reply.planning, valid.includes(planning) ? planning : undefined);
    assert.deepEqual(history[0].planning, reply.planning);
    assert.deepEqual(history[0].feedback, feedback);
    assert.equal(history[1].answerDeleted, true);
    assert.equal(seen.length, valid.includes(planning) ? 1 : 0);
  }
});

test("feedback create/change/cancel uses real server message ID and survives history restoration", async () => {
  saveMemberTokens("access-token", "refresh-token");
  let feedback = null;
  const calls = [];
  global.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    if (init.method === "PUT") {
      const body = JSON.parse(init.body);
      assert.equal(body.message_id, ASSISTANT_MSG);
      feedback = body.rating === null ? null : { rating: body.rating, reason: body.reason, comment: body.comment };
      return json({ feedback });
    }
    return json([{ ...row(ASSISTANT_MSG, "assistant", "answer"), feedback }]);
  };
  assert.deepEqual(await saveAnswerFeedback("member", SESSION, ASSISTANT_MSG, { rating: "up", reason: "", comment: "" }), { rating: "up", reason: "", comment: "" });
  await saveAnswerFeedback("member", SESSION, ASSISTANT_MSG, { rating: "down", reason: "incorrect", comment: "wrong" });
  assert.deepEqual(restoreChatMessages(await fetchChatHistory("member", SESSION))[0].feedback, feedback);
  assert.equal(await saveAnswerFeedback("member", SESSION, ASSISTANT_MSG, null), null);
  assert.equal(calls[0].url, `/api/v2/chat/sessions/${SESSION}/feedback/`);
  assert.equal(new Headers(calls[0].init.headers).get("Authorization"), "Bearer access-token");
  await saveAnswerFeedback("guest", SESSION, ASSISTANT_MSG, { rating: "up", reason: "", comment: "" });
  assert.equal(new Headers(calls.at(-1).init.headers).get("Authorization"), null);
  assert.equal(calls.at(-1).init.credentials, undefined);
});

test("feedback validates IDs, comment boundaries and API errors without optimistic success", async () => {
  global.fetch = async () => json({ detail: "stale answer" }, 404);
  await assert.rejects(saveAnswerFeedback("guest", SESSION, 0, null), error => error.status === 400);
  await assert.rejects(saveAnswerFeedback("guest", SESSION, 2, { rating: "down", reason: "other", comment: "x".repeat(1001) }), error => error.status === 400);
  await assert.rejects(saveAnswerFeedback("guest", SESSION, 2, null), error => error.status === 404);
  global.fetch = async () => json({ feedback: { rating: "invented" } });
  await assert.rejects(saveAnswerFeedback("guest", SESSION, 2, null), error => error.status === 502);
});

test("admin feedback list filters/page and detail reuse Bearer requests", async () => {
  saveMemberTokens("access-token", "refresh-token");
  const detail = { id: 1, session_id: SESSION, answer_id: "stored-answer-id", message_id: 2, rating: "down", reason: "other", comment: "", question: "q", answer: "a", metadata: {}, created_at: "now", updated_at: "now" };
  global.fetch = async (url, init) => {
    assert.equal(new Headers(init.headers).get("Authorization"), "Bearer access-token");
    if (String(url).includes("?")) {
      assert.equal(String(url), "/api/v2/chat/admin/feedback/?page=2&rating=down&reason=other");
      return json({ count: 21, results: [detail] });
    }
    assert.equal(String(url), "/api/v2/chat/admin/feedback/1/");
    return json(detail);
  };
  assert.equal((await fetchAdminFeedback(2, "down", "other")).count, 21);
  assert.deepEqual(await fetchAdminFeedbackDetail(1), detail);
});
