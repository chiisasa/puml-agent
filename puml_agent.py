import os, sys, subprocess, webbrowser
import json
import base64
import shutil
import datetime
from pathlib import Path
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import gradio as gr

from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from openai import OpenAI

class SWInput(BaseModel):
    sw_input: str = Field(..., description="PlantUML code block")

class Review(BaseModel):
    review_passed: bool = Field(..., description="Whether the generated PlantUML diagram passed the review")
    improvement_points: list[str] = Field(..., description="List of improvement points if the review did not pass")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    requirements_txt: str
    requirements_attachments: list[str]
    attempt_num: int
    sw_input_gen: bool
    sw_input: str
    sw_exec: str
    sw_output: str
    review_gen: bool
    review_passed: bool
    improvement_points: list[str]

class PlantUMLAgent:
    def __init__(self, plantuml_path: str = "./plantuml", max_iterations: int = 3, use_web: bool = False):
        print("Agent initializing...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.oclient = OpenAI()
        self.llm = ChatOpenAI(model="gpt-5", output_version="responses/v1") # gpt-4.1 | gpt-5
        self.llm_response_sw_input = self.llm.with_structured_output(SWInput)
        self.llm_response_review = self.llm.with_structured_output(Review)

        self.use_web = use_web
        self.tools = [{"type":"web_search"}] if use_web else []
        self.plantuml_path = Path(plantuml_path).resolve()
        self.max_iterations = max_iterations
        self.tmp_dir = Path("./tmp").resolve()
        self.tmp_dir.mkdir(exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.tmp_dir / f"plantuml_agent_log_{ts}.json"

        self.checkpointer = InMemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        try:
            out_path = self.tmp_dir / "workflow_graph.png"
            self.app.get_graph().draw_mermaid_png(output_file_path=str(out_path))
            self._open_image(out_path)
        except Exception as e:
            print(f"Error generating workflow graph: {e}")

    def _build_workflow(self) -> StateGraph:
        g = StateGraph(State)
        g.add_node("build_software", self.build_plantuml_if_needed)
        g.add_node("generate_input", self.generate_plantuml_code)
        g.add_node("execute_software", self.execute_plantuml)
        g.add_node("review_output", self.review_plantuml_diagram)

        g.add_edge(START, "build_software")
        g.add_edge("build_software", "generate_input")
        g.add_conditional_edges(
            "generate_input",
            self.should_retry_generation,
            {
                "failed": "generate_input",
                "succeeded": "execute_software",
                "max_attempts_reached": END
            },
        )
        g.add_edge("execute_software", "review_output")
        g.add_conditional_edges(
            "review_output",
            self.should_retry,
            {
                "needs_improvement": "generate_input",
                "passed_OR_max_attempts_reached": END
            }
        )
        return g
    
    # --- nodes ---
    def build_plantuml_if_needed(self, state: State) -> dict:
        print("Checking if PlantUML JAR needs to be built...")
        build_libs_dir = self.plantuml_path / "build" / "libs"
        jar = self._find_plantuml_jar(build_libs_dir)
        if not jar:
            try:
                gradle_cmd = str((self.plantuml_path / ("gradlew.bat" if sys.platform == "win32" else "gradlew")).resolve())
                subprocess.run([gradle_cmd, "build", "-x", "test"], cwd=self.plantuml_path, check=True, timeout=600)
            except Exception as e:
                print(f"Error building PlantUML: {e}")
            jar = self._find_plantuml_jar(build_libs_dir)

        if not jar:
            raise FileNotFoundError(f"PlantUML JAR file not found in {self.plantuml_path}")
        return {"messages": [SystemMessage(content=f"[OBSERVATION] PlantUML JAR: {jar}")], "sw_exec": jar}

    def generate_plantuml_code(self, state: State) -> dict:
        attempt = state.get("attempt_num", 0) + 1
        print(f"\n🚀 Attempt {attempt} to generate PlantUML code...")

        system_msg = SystemMessage(content="あなたはPlantUML図の生成者です。要求に基づいて正確なPlantUMLコードを生成してください。")
        prompt = f"""\
以下の要求に基づいて、動作するPlantUMLコードを生成してください。
前回のPlantUMLコードに対する改善点があれば、それを反映したコードを生成してください。

要求: {state['requirements_txt']}
添付ファイル: {', '.join(state.get('requirements_attachments', [])) if state.get('requirements_attachments') else 'なし'}

前回のPlantUMLコードに対する改善点: 
{chr(10).join(state.get('improvement_points', [])) if state.get('improvement_points') else 'なし'}

望ましい出力形式:
PlantUMLコード: [PlantUMLコード]
簡易説明: [PlantUMLコードの簡易説明または改善箇所の説明]
"""
        user_msg = HumanMessage(content=prompt)
        tools = self._tools_with_files([], state.get("requirements_attachments"))
        
        try:
            ai = self.llm.bind_tools(tools, tool_choice="auto").invoke(self._keep_only_last_image(state["messages"] + [system_msg, user_msg]))
            self._log_web_search_results(ai)
            sw_input = self.llm_response_sw_input.invoke([ai]).sw_input
            return {"messages": [system_msg, user_msg, ai], "sw_input": sw_input, "attempt_num": attempt, "sw_input_gen": True}
        except Exception as e:
            print(f"PlantUML code generation error: {e}")
            return {
                "messages": [user_msg, SystemMessage(content=f"[SYSTEM ERROR] PlantUML生成失敗: {e}")],
                "sw_input": "",
                "attempt_num": attempt,
                "sw_input_gen": False,
            }

    def should_retry_generation(self, state: State) -> Literal["failed", "succeeded", "max_attempts_reached"]:
        attempt_num = state.get("attempt_num", 0)
        sw_input_gen = state.get("sw_input_gen", False)

        if attempt_num >= self.max_iterations:
            return "max_attempts_reached" if not sw_input_gen else "succeeded"

        return "failed" if not sw_input_gen else "succeeded"
    
    def execute_plantuml(self, state: State) -> dict:
        print("Executing PlantUML...")
        code, jar = state.get("sw_input", ""), state.get("sw_exec", "")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        puml = self.tmp_dir / f"plantuml_diagram_{ts}.puml"
        png = self.tmp_dir / f"plantuml_diagram_{ts}.png"
        puml.write_text(code, encoding="utf-8")

        try:
            r = subprocess.run(["java", "-DPLANTUML_LIMIT_SIZE=8192", "-jar", jar, "-tpng", str(puml)], capture_output=True, text=True, timeout=30)
            temp_png = puml.with_suffix(".png")
            if temp_png.exists():
                shutil.move(str(temp_png), str(png))
                self._open_image(png)
                msg = f"PlantUML実行成功: {png.name}"
                ok = True
            else:
                msg = f"PlantUML実行失敗: {r.stderr.strip()}"
                ok = False
        except Exception as e:
            print(f"PlantUML execution error: {e}")
            msg, ok = f"PlantUML実行エラー: {e}", False

        return {"messages": [SystemMessage(content=f"[OBSERVATION] {msg}")], "sw_output": str(png if ok else "")}

    def review_plantuml_diagram(self, state: State) -> dict:
        print("Generating review...")
        out = state.get("sw_output", "")
        if not out:
            obs = "画像未生成"
            return {"messages": [SystemMessage(content=f"[OBSERVATION] {obs}")], "review_gen": False, "review_passed": False}

        try:
            b64 = base64.b64encode(Path(out).read_bytes()).decode("utf-8")
        except Exception as e:
            print(f"Image read error: {e}")
            obs = f"画像読込失敗: {e}"
            return {"messages": [SystemMessage(content=f"[OBSERVATION] {obs}")], "review_gen": False, "review_passed": False}

        system_msg = SystemMessage(content="あなたはPlantUML図の評価者です。要求適合性を評価してください。")
        use_web_prompt = "Web検索ツールを使用して情報を取得し、評価に反映してください。" if self.use_web else ""
        review_prompt = f"""\
要求に対して、以下の図を生成しました。
元の要求: {state['requirements_txt']}
添付ファイル: {', '.join(state.get('requirements_attachments', [])) if state.get('requirements_attachments') else 'なし'}

図の内容: 
[画像を参照]

以下の観点で評価してください。{use_web_prompt}

評価観点: 
- 要求を全て満たしているか
- 要求に対して過剰な要素が含まれていないか

望ましい出力形式:
評価: [満足/要改善]
原因: [要求を満たさない原因や、過剰な要素を特定し、具体的に説明]
PlantUMLコードの改善点: [特定した原因を除去する方法を、可能な限り詳細に説明]
"""
        analysis = HumanMessage(content=[{"type": "input_text", "text": review_prompt}, {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}])
        tools = self._tools_with_files(self.tools, state.get("requirements_attachments"))

        try:
            ai = self.llm.bind_tools(tools, tool_choice="auto").invoke(self._keep_only_last_image(state["messages"] + [system_msg, analysis]))
            self._log_web_search_results(ai)
            review = self.llm_response_review.invoke([ai])
            print("Improvement points:\n" + "\n".join(f"- {p}" for p in (review.improvement_points or ["なし"])))
            return {"messages": [system_msg, analysis, ai], "review_gen": True, "review_passed": review.review_passed, "improvement_points": review.improvement_points}
        except Exception as e:
            print(f"Review generation error: {e}")
            obs = f"レビュー失敗: {e}"
            return {"messages": [system_msg, analysis, SystemMessage(content=f"[OBSERVATION] {obs}")], "review_gen": False, "review_passed": False}

    def should_retry(self, state: State) -> Literal["needs_improvement", "passed_OR_max_attempts_reached"]:
        if state.get("review_passed", False):
            return "passed_OR_max_attempts_reached"
        if state.get("attempt_num", 0) >= self.max_iterations:
            return "passed_OR_max_attempts_reached"
        return "needs_improvement"

    # --- utils ---
    def _tools_with_files(self, tools: list[str], file_ids: list[str]):
        return tools if not file_ids else tools + [{"type":"code_interpreter","container":{"type":"auto","file_ids":file_ids}}]
    
    def _find_plantuml_jar(self, build_libs_dir: Path) -> str:
        if not build_libs_dir.exists():
            return ""
        jars = sorted(
            [p for p in build_libs_dir.glob("plantuml*.jar") if not (str(p).endswith("-sources.jar") or str(p).endswith("-javadoc.jar"))],
            key=lambda p: p.name,
            reverse=True,
        )
        return str(jars[0]) if jars else ""

    def _open_image(self, p: Path):
      p = str(Path(p).resolve())
      try:
          {"win32": lambda x: os.startfile(x),
          "darwin": lambda x: subprocess.run(["open", x], check=False)
          }.get(sys.platform, lambda x: subprocess.run(["xdg-open", x], check=False))(p)
      except Exception as e:
          try:
              webbrowser.open(Path(p).as_uri())
          except Exception as e2:
              print(f"[WARN] open failed: {e} / browser: {e2}")

    def _keep_only_last_image(self, msgs: list[BaseMessage]) -> list[BaseMessage]:
        found, out = False, []
        for m in reversed(msgs):
            if not isinstance(m.content, list):
                out.append(m)
                continue
            parts = []
            for p in m.content:
                if isinstance(p, dict) and p.get("type") == "input_image":
                    p = p if not found else {"type": "input_text", "text": "[image removed]"}
                    found = True
                parts.append(p)
            out.append(m.__class__(content=parts, name=getattr(m, "name", None)))
        return list(reversed(out))
    
    # --- logging ---
    def _log_web_search_results(self, ai: AIMessage):
        c = ai.content
        qs = [p['action']['query'] for p in c if p['type'] == 'web_search_call']
        cs = [(a['title'], a['url']) for p in c for a in p.get('annotations', [])]
        if qs or cs: 
            print("\n".join((["Web Queries:"]+[f"- {q}" for q in qs] if qs else []) + (["Citations:"]+[f"- {t} | {u}" for t,u in cs] if cs else [])))

    def _save_messages_to_log_file(self, result: dict, thread_id: str):
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "thread_id": thread_id,
            "messages": [{"type": (d:=m.model_dump()).pop("type"), **d} for m in result.get("messages", [])],
            "final_state": {
                "review_gen": result.get("review_gen", False),
                "review_passed": result.get("review_passed", False),
                "attempt_num": result.get("attempt_num", 0),
                "sw_output": result.get("sw_output", ""),
                "sw_input_gen": result.get("sw_input_gen", False),
            },
        }
        try:
            self.log_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Error saving log file: {e}")

    # --- API ---
    def invoke(
        self, 
        requirements_txt: str, 
        requirements_attachments: list[str], 
        thread_id: str = "default"
    ) -> str:
        file_ids = []
        for p in requirements_attachments or []:
            try:
                with open(p, "rb") as f:
                    up = self.oclient.files.create(file=f, purpose="user_data", expires_after={"anchor": "created_at", "seconds": 86400})
                file_ids.append(up.id)
            except Exception as e:
                print(f"[WARN] file upload failed: {p} ({e})")

        config = {"configurable": {"thread_id": thread_id}}
        init = {
            "messages": [HumanMessage(content=requirements_txt)],
            "requirements_txt": requirements_txt,
            "requirements_attachments": file_ids,
            "attempt_num": 0,
            "sw_input": "",
            "sw_output": "",
            "review_gen": False,
            "review_passed": False,
            "sw_input_gen": False,
            "sw_exec": "",
        }
        try:
            result = self.app.invoke(init, config)
            self._save_messages_to_log_file(result, thread_id)
        except Exception as e:
            return f"Error occurred: {e}"

        if result.get("review_passed"):
            return "Review passed."

        attempt = result.get("attempt_num", 0)
        if not result.get("sw_input_gen", False):
            return "PlantUML Code generation failed."
        if not result.get("review_gen", False):
            return "Review generation failed."
        if attempt >= self.max_iterations:
            msg = f"{attempt} attempts made but failed to generate a valid diagram."
            return msg
        return "PlantUML code generation failed. Please try again."

def main():
    plantuml_path = input("Path to PlantUML project [./plantuml]: ").strip() or "./plantuml"
    try:
        max_iterations = int(input("Maximum code generation attempts [3]: ").strip() or "3")
    except ValueError:
        max_iterations = 3
    use_web = input("Use web search? [y/N]: ").strip().lower() in {"y", "yes"}

    agent = PlantUMLAgent(plantuml_path, max_iterations, use_web)
    thread_id = "plantuml_session"

    def run(requirements_txt, requirements_attachments):
        res = agent.invoke(requirements_txt or "", requirements_attachments or [], thread_id)
        print(f"\n{res}")
        return res

    gr.Interface(
        fn=run,
        inputs=[
            gr.Textbox(lines=12, label="要求"),
            gr.Files(type="filepath", label="添付ファイル（任意・複数可）"),
        ],
        outputs=[gr.Textbox(lines=1, label="結果")],
        title="PlantUML Agent", flagging_mode="never"
    ).launch()

if __name__ == "__main__":
    main()
