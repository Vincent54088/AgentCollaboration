"""Plan and Solve Agent实现 - 分解规划与逐步执行的智能体"""

import json
from typing import Optional, List, Dict, TYPE_CHECKING, Any, AsyncGenerator

from ..core.agent import Agent
from ..core.llm import AgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..core.streaming import StreamEvent, StreamEventType
from ..core.lifecycle import LifecycleHook
from ..core.plan_checkpoint import (
    PlanCheckpointStore,
    STEP_PENDING,
    STEP_IN_PROGRESS,
    STEP_COMPLETED,
)

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

class Planner:
    """规划器 - 负责将复杂问题分解为简单步骤（使用 Function Calling）"""

    def __init__(self, llm_client: AgentsLLM, system_prompt: Optional[str] = None):
        self.llm_client = llm_client
        self.system_prompt = system_prompt or """你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。"""

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划（使用 Function Calling）

        Args:
            question: 要解决的问题
            **kwargs: LLM调用参数

        Returns:
            步骤列表
        """
        print("--- 正在生成计划 ---")

        # 定义计划生成工具
        plan_tool = {
            "type": "function",
            "function": {
                "name": "generate_plan",
                "description": "生成解决问题的分步计划",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "按顺序排列的执行步骤列表"
                        }
                    },
                    "required": ["steps"]
                }
            }
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请为以下问题生成详细的执行计划：\n\n{question}"}
        ]

        try:
            response = self.llm_client.invoke_with_tools(
                messages=messages,
                tools=[plan_tool],
                tool_choice={"type": "function", "function": {"name": "generate_plan"}},
                **kwargs
            )

            # 提取工具调用结果
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                arguments = json.loads(tool_call.arguments)
                plan = arguments.get("steps", [])

                print(f"✅ 计划已生成:")
                for i, step in enumerate(plan, 1):
                    print(f"  {i}. {step}")

                return plan
            else:
                print("❌ 模型未返回计划工具调用")
                return []

        except Exception as e:
            print(f"❌ 生成计划时发生错误: {e}")
            return []

class Executor:
    """执行器 - 负责按计划逐步执行（支持 Function Calling）"""

    def __init__(
        self,
        llm_client: AgentsLLM,
        system_prompt: Optional[str] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3
    ):
        self.llm_client = llm_client
        self.system_prompt = system_prompt or """你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
请专注于解决当前步骤，并输出该步骤的最终答案。"""
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.max_tool_iterations = max_tool_iterations

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按计划执行任务（支持 Function Calling）

        Args:
            question: 原始问题
            plan: 执行计划
            **kwargs: LLM调用参数

        Returns:
            最终答案
        """
        history = []
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")

            # 构建上下文消息
            context = f"""# 原始问题:
{question}

# 完整计划:
{self._format_plan(plan)}

# 历史步骤与结果:
{self._format_history(history) if history else "无"}

# 当前步骤:
{step}

请执行当前步骤并给出结果。"""

            # 执行单个步骤（支持工具调用）
            response_text = self._execute_step(context, **kwargs)

            history.append({"step": step, "result": response_text})
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")

        return final_answer

    def _format_plan(self, plan: List[str]) -> str:
        """格式化计划列表"""
        return "\n".join([f"{i}. {step}" for i, step in enumerate(plan, 1)])

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """格式化历史记录"""
        return "\n\n".join([f"步骤 {i}: {h['step']}\n结果: {h['result']}"
                           for i, h in enumerate(history, 1)])

    def _execute_step(self, context: str, **kwargs) -> str:
        """
        执行单个步骤（支持 Function Calling）

        Args:
            context: 上下文信息
            **kwargs: 其他参数

        Returns:
            步骤执行结果
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]

        # 如果没有启用工具调用，直接返回
        if not self.enable_tool_calling or not self.tool_registry:
            llm_response = self.llm_client.invoke(messages, **kwargs)
            return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        # 启用工具调用模式
        from .simple_agent import SimpleAgent
        # 临时创建一个 SimpleAgent 实例来复用工具调用逻辑
        temp_agent = SimpleAgent(
            name="temp_executor",
            llm=self.llm_client,
            tool_registry=self.tool_registry
        )
        tool_schemas = temp_agent._build_tool_schemas()

        current_iteration = 0

        while current_iteration < self.max_tool_iterations:
            current_iteration += 1

            try:
                response = self.llm_client.invoke_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    **kwargs
                )
            except Exception as e:
                print(f"❌ LLM 调用失败: {e}")
                break

            # 处理工具调用
            tool_calls = response.tool_calls
            if not tool_calls:
                # 没有工具调用，返回文本响应
                return response.content or ""

            # 将助手消息添加到历史
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # 执行所有工具调用
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_call_id = tool_call.id

                try:
                    arguments = json.loads(tool_call.arguments)
                except json.JSONDecodeError as e:
                    print(f"❌ 工具参数解析失败: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"错误：参数格式不正确 - {str(e)}"
                    })
                    continue

                # 执行工具（复用基类方法）
                result = temp_agent._execute_tool_call(tool_name, arguments)

                # 添加工具结果到消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                })

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= self.max_tool_iterations:
            llm_response = self.llm_client.invoke(messages, **kwargs)
            return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        return ""

class PlanSolveAgent(Agent):
    """
    Plan and Solve Agent - 分解规划与逐步执行的智能体

    这个Agent能够：
    1. 将复杂问题分解为简单步骤（使用 Function Calling）
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案
    5. 支持工具调用（可选）

    特别适合多步骤推理、数学问题、复杂分析等任务。
    """

    def __init__(
        self,
        name: str,
        llm: AgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        planner_prompt: Optional[str] = None,
        executor_prompt: Optional[str] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3
    ):
        """
        初始化PlanSolveAgent

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词（Agent级别）
            config: 配置对象
            planner_prompt: 规划器的系统提示词（可选）
            executor_prompt: 执行器的系统提示词（可选）
            tool_registry: 工具注册表（可选）
            enable_tool_calling: 是否启用工具调用
            max_tool_iterations: 最大工具调用迭代次数
        """
        # 传递 tool_registry 到基类
        super().__init__(
            name,
            llm,
            system_prompt,
            config,
            tool_registry=tool_registry
        )

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(
            self.llm,
            executor_prompt,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
            max_tool_iterations=max_tool_iterations
        )

        # 计划检查点（中断恢复）
        self._session_id = self.trace_logger.session_id if self.trace_logger else self._generate_session_id()
        self._checkpoint_store: Optional[PlanCheckpointStore] = None
        if self.config.plan_checkpoint_enabled:
            self._checkpoint_store = PlanCheckpointStore(
                session_id=self._session_id,
                checkpoint_dir=self.config.plan_checkpoint_dir
            )
    
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Plan and Solve Agent

        Args:
            input_text: 要解决的问题
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        # 执行状态（供中断时保存检查点使用）
        plan: List[str] = []
        statuses: List[str] = []
        results: List[str] = []

        try:
            # 1. 生成计划
            plan = self.planner.plan(input_text, **kwargs)
            if not plan:
                final_answer = "无法生成有效的行动计划，任务终止。"
                print(f"\n--- 任务终止 ---\n{final_answer}")

                # 保存到历史记录
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))

                return final_answer

            # 2. 执行计划（逐步写检查点）
            statuses = [STEP_PENDING] * len(plan)
            results = [""] * len(plan)

            final_answer = self._run_steps(
                input_text, plan, statuses, results,
                start_idx=0, recovery_prompt="", **kwargs
            )
            print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

            # 全部完成，清理检查点
            self._clear_checkpoint()

            # 保存到历史记录
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))

            return final_answer

        except KeyboardInterrupt:
            # Ctrl+C 时保存检查点（与 react_agent 的中断处理一致）
            print("\n⚠️ 用户中断，自动保存计划检查点...")
            self._save_checkpoint_silent(input_text, plan, statuses, results, interrupted=True)
            raise

        except Exception as e:
            # 错误时也保存检查点
            print(f"\n❌ 发生错误: {e}，自动保存计划检查点...")
            self._save_checkpoint_silent(input_text, plan, statuses, results, interrupted=True)
            raise

    # ==================== 检查点持久化 ====================

    def _save_checkpoint_silent(
        self,
        question: str,
        plan: List[str],
        statuses: List[str],
        results: List[str],
        interrupted: bool
    ) -> None:
        """静默保存检查点（失败不影响主流程）"""
        if not plan:
            return

        try:
            self._checkpoint(
                question, plan, statuses, results,
                current_step_index=self._current_step_index(plan, statuses),
                interrupted=interrupted
            )
            if self.config.debug:
                print(f"✅ 计划检查点已保存: {self._checkpoint_store.filepath}")
        except Exception as e:
            if self.config.debug:
                print(f"⚠️ 检查点保存失败: {e}")

    def _checkpoint(
        self,
        question: str,
        plan: List[str],
        statuses: List[str],
        results: List[str],
        current_step_index: int,
        interrupted: bool
    ) -> None:
        """写入计划检查点（静默失败）"""
        if not self._checkpoint_store:
            return

        try:
            self._checkpoint_store.save(
                session_id=self._session_id,
                agent_name=self.name,
                question=question,
                plan=plan,
                step_statuses=statuses,
                step_results=results,
                current_step_index=current_step_index,
                interrupted=interrupted,
            )
        except Exception as e:
            if self.config.debug:
                print(f"⚠️ 检查点写入失败: {e}")

    def _clear_checkpoint(self) -> None:
        """清理检查点文件（全部步骤完成时调用）"""
        if not self._checkpoint_store:
            return

        try:
            self._checkpoint_store.clear()
        except Exception:
            pass

    @staticmethod
    def _current_step_index(plan: List[str], statuses: List[str]) -> int:
        """计算当前步骤索引（第一个未完成步骤；全部完成则为最后一步）"""
        for i, status in enumerate(statuses):
            if status != STEP_COMPLETED:
                return i
        return max(len(plan) - 1, 0)

    # ==================== 步骤执行循环 ====================

    def _run_steps(
        self,
        question: str,
        plan: List[str],
        statuses: List[str],
        results: List[str],
        start_idx: int,
        recovery_prompt: str,
        **kwargs
    ) -> str:
        """从 start_idx 开始顺序执行计划步骤，每步实时写检查点

        Args:
            question: 原始问题
            plan: 计划步骤列表
            statuses: 每步状态列表（就地更新）
            results: 每步结果列表（就地更新）
            start_idx: 起始步骤索引（0 起）
            recovery_prompt: 断点恢复提示段（正常执行时为空字符串）
            **kwargs: LLM 调用参数

        Returns:
            最后一步的执行结果
        """
        final_answer = ""
        total = len(plan)

        for i in range(start_idx, total):
            step = plan[i]
            step_num = i + 1
            print(f"\n-> 正在执行步骤 {step_num}/{total}: {step}")

            # 标记进行中并写检查点
            statuses[i] = STEP_IN_PROGRESS
            self._checkpoint(question, plan, statuses, results, current_step_index=i, interrupted=False)

            # 构建上下文并执行单个步骤（复用 Executor 的工具调用逻辑）
            context = self._build_step_context(question, plan, statuses, results, i, recovery_prompt)
            response_text = self.executor._execute_step(context, **kwargs)

            # 标记完成并写检查点
            statuses[i] = STEP_COMPLETED
            results[i] = response_text
            self._checkpoint(question, plan, statuses, results, current_step_index=i, interrupted=False)

            print(f"✅ 步骤 {step_num} 已完成")
            final_answer = response_text

        return final_answer

    def _build_step_context(
        self,
        question: str,
        plan: List[str],
        statuses: List[str],
        results: List[str],
        current_idx: int,
        recovery_prompt: str
    ) -> str:
        """构建单个步骤的执行上下文

        正常执行时与 Executor.execute 的上下文格式保持一致；
        恢复执行时在顶部注入断点恢复提示段。
        """
        # 已完成步骤（含结果）
        history_lines = []
        for j in range(current_idx):
            if statuses[j] == STEP_COMPLETED:
                history_lines.append(f"步骤 {j+1}: {plan[j]}\n结果: {results[j]}")
        history_text = "\n\n".join(history_lines) if history_lines else "无"

        plan_text = "\n".join([f"{k+1}. {s}" for k, s in enumerate(plan)])

        context = f"""# 原始问题:
{question}

# 完整计划:
{plan_text}

# 历史步骤与结果:
{history_text}

# 当前步骤:
{plan[current_idx]}"""

        if recovery_prompt:
            return f"{recovery_prompt}\n\n{context}"
        return context

    # ==================== 断点恢复 ====================

    def _build_recovery_prompt(
        self,
        question: str,
        plan: List[str],
        statuses: List[str],
        results: List[str],
        interrupted_idx: int,
        new_context: Optional[str] = None
    ) -> str:
        """构建断点恢复提示段

        参考 Claude Code 的 Plan Recovery System Prompt：
        明确告知模型已完成步骤、被中断步骤、待执行步骤，
        并要求先核验环境落盘状态再继续执行。
        """
        # 已完成步骤
        completed_lines = []
        for j in range(interrupted_idx):
            if statuses[j] == STEP_COMPLETED:
                completed_lines.append(f"步骤 {j+1}: {plan[j]}\n结果: {results[j]}")
        completed_text = "\n".join(completed_lines) if completed_lines else "无"

        # 待执行步骤（含被中断步骤）
        pending_lines = [f"步骤 {k+1}: {plan[k]}" for k in range(interrupted_idx, len(plan))]
        pending_text = "\n".join(pending_lines) if pending_lines else "无"

        interrupted_text = plan[interrupted_idx] if interrupted_idx < len(plan) else "（无）"

        # 环境真相核验（subprocess 自动检查 git 状态）
        env_summary = self._verify_environment()

        prompt = f"""## 断点恢复（中断后续执行）
本次执行曾在步骤 {interrupted_idx + 1} 处被中断。当前状态：
- 已完成步骤（1..{interrupted_idx}）：
{completed_text}

- 被中断步骤（{interrupted_idx + 1}）：{interrupted_text}

- 待执行步骤（{interrupted_idx + 2}..{len(plan)}）：
{pending_text}

【环境核验】恢复时已自动检查代码库状态：{env_summary}
请先确认已完成步骤的产物确实存在于磁盘（必要时用 Read 工具复核相关文件）；
若被中断步骤的实际改动已落盘，可直接跳到下一步。
确认后从步骤 {interrupted_idx + 1} 继续执行。"""

        if new_context:
            prompt += f"""

【用户新增上下文】
{new_context}
请将以上新要求与原始计划合并后继续执行。"""

        return prompt

    def _verify_environment(self) -> str:
        """自动核验代码库落盘状态（subprocess 运行 git status / git diff）

        非 git 仓库或命令失败时静默降级，不影响恢复流程。

        Returns:
            环境核验摘要文本
        """
        import subprocess

        try:
            # 检查是否在 git 仓库中
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return "（当前目录不是 git 仓库，跳过自动核验）"

            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10
            )
            diff = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True, text=True, timeout=10
            )

            lines = []
            changed_lines = [ln for ln in status.stdout.splitlines() if ln.strip()]
            if changed_lines:
                lines.append(f"已变更文件 {len(changed_lines)} 个：")
                lines.extend(changed_lines[:20])
            else:
                lines.append("工作区干净（无未提交变更）")

            if diff.stdout.strip():
                lines.append(f"未暂存 diff 摘要：\n{diff.stdout.strip()[:2000]}")

            return "\n".join(lines)
        except Exception:
            return "（环境核验失败，已跳过）"

    def resume(
        self,
        question: Optional[str] = None,
        resume_file: Optional[str] = None,
        new_context: Optional[str] = None,
        **kwargs
    ) -> str:
        """从检查点恢复执行被中断的计划

        Args:
            question: 新的问题描述（可选，与检查点不一致时仅提示并合并上下文）
            resume_file: 显式指定检查点文件路径（可选）
            new_context: 用户新增上下文（可选，合并进恢复提示词）
            **kwargs: LLM 调用参数

        Returns:
            最终答案

        Raises:
            RuntimeError: 计划检查点未启用
            FileNotFoundError: 未找到可恢复的检查点
        """
        if not self._checkpoint_store:
            raise RuntimeError("计划检查点未启用（plan_checkpoint_enabled=False），无法恢复")

        # 1. 加载检查点
        if resume_file:
            data = PlanCheckpointStore.load_file(resume_file)
        else:
            data = PlanCheckpointStore.latest(self.config.plan_checkpoint_dir)

        if not data:
            raise FileNotFoundError(
                "未找到可恢复的计划检查点。请先运行任务（run），或通过 resume_file 指定检查点文件路径。"
            )

        plan = data.get("plan", [])
        statuses = data.get("step_statuses", [])
        results = data.get("step_results", [])
        saved_question = data.get("question", "")

        if not plan:
            print("⚠️ 检查点中计划为空，无法恢复。")
            return "无法恢复：检查点中无有效计划。"

        if question and saved_question and question != saved_question:
            print("⚠️ 新问题与检查点原始问题不一致，将合并新上下文继续执行。")

        # 2. 确定中断位置（第一个未完成步骤）
        interrupted_idx = 0
        for i, status in enumerate(statuses):
            if status != STEP_COMPLETED:
                interrupted_idx = i
                break
        else:
            interrupted_idx = max(len(plan) - 1, 0)

        # 全部步骤均已完成的检查点（理论不会出现，防御处理）
        if all(s == STEP_COMPLETED for s in statuses):
            print("✅ 检查点中所有步骤均已完成。")
            return results[-1] if results else ""

        # 3. 构建恢复提示词并继续执行
        # 记录恢复来源的会话 ID（用于完成时清理该检查点文件）
        resumed_session_id = data.get("session_id", "")

        recovery_prompt = self._build_recovery_prompt(
            saved_question, plan, statuses, results, interrupted_idx, new_context
        )

        print(f"\n--- 断点恢复：从步骤 {interrupted_idx + 1}/{len(plan)} 继续执行 ---")

        final_answer = self._run_steps(
            saved_question, plan, statuses, results,
            start_idx=interrupted_idx, recovery_prompt=recovery_prompt, **kwargs
        )

        # 全部完成，清理检查点（包括恢复来源文件，可能属于其他会话）
        self._clear_checkpoint()
        if resumed_session_id and resumed_session_id != self._session_id:
            try:
                source_store = PlanCheckpointStore(
                    session_id=resumed_session_id,
                    checkpoint_dir=self.config.plan_checkpoint_dir
                )
                source_store.clear()
            except Exception:
                pass

        self.add_message(Message(saved_question, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    async def arun_stream(
        self,
        input_text: str,
        on_start: LifecycleHook = None,
        on_finish: LifecycleHook = None,
        on_error: LifecycleHook = None,
        resume_checkpoint: Optional[Dict[str, Any]] = None,
        new_context: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        PlanAgent 真正的流式执行

        实时返回：
        - 规划阶段的计划生成
        - 执行阶段的每个步骤输出
        - 恢复执行（resume_checkpoint）时返回断点恢复事件

        Args:
            input_text: 用户输入
            on_start: 开始钩子
            on_finish: 完成钩子
            on_error: 错误钩子
            resume_checkpoint: 检查点字典或文件路径（可选）。None 时自动检测
               最新未完成检查点（若存在则进入恢复执行）
            new_context: 用户新增上下文（可选，恢复时合并）
            **kwargs: 其他参数

        Yields:
            StreamEvent: 流式事件
        """
        # 发送开始事件
        yield StreamEvent.create(
            StreamEventType.AGENT_START,
            self.name,
            input_text=input_text
        )

        # 执行状态（供检查点使用）
        plan: List[str] = []
        statuses: List[str] = []
        results: List[str] = []
        saved_question = input_text
        recovery_prompt = ""

        try:
            # ---- 检查是否有可恢复的检查点 ----
            checkpoint_data = None
            if resume_checkpoint is not None:
                if isinstance(resume_checkpoint, str):
                    checkpoint_data = PlanCheckpointStore.load_file(resume_checkpoint)
                else:
                    checkpoint_data = resume_checkpoint
            elif self._checkpoint_store:
                checkpoint_data = PlanCheckpointStore.latest(self.config.plan_checkpoint_dir)

            if checkpoint_data:
                # ---- 恢复执行路径 ----
                plan = checkpoint_data.get("plan", [])
                statuses = checkpoint_data.get("step_statuses", [])
                results = checkpoint_data.get("step_results", [])
                saved_question = checkpoint_data.get("question", input_text)
                resumed_session_id = checkpoint_data.get("session_id", "")

                if not plan:
                    error_msg = "检查点中计划为空，无法恢复。"
                    yield StreamEvent.create(
                        StreamEventType.ERROR, self.name, error=error_msg, phase="recovery"
                    )
                    yield StreamEvent.create(StreamEventType.AGENT_FINISH, self.name, result=error_msg)
                    return

                interrupted_idx = 0
                for i, s in enumerate(statuses):
                    if s != STEP_COMPLETED:
                        interrupted_idx = i
                        break

                recovery_prompt = self._build_recovery_prompt(
                    saved_question, plan, statuses, results, interrupted_idx, new_context
                )

                print(f"\n--- 断点恢复：从步骤 {interrupted_idx + 1}/{len(plan)} 继续执行 ---")
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    phase="recovery",
                    resumed=True,
                    description=f"检测到中断，从步骤 {interrupted_idx + 1} 恢复执行"
                )
            else:
                # ---- 正常执行路径：规划 ----
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    phase="planning",
                    description="生成执行计划"
                )

                print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

                # 生成计划（同步方法，暂时保持）
                plan = self.planner.plan(input_text, **kwargs)

                if not plan:
                    error_msg = "无法生成有效的行动计划，任务终止。"

                    yield StreamEvent.create(
                        StreamEventType.ERROR,
                        self.name,
                        error=error_msg,
                        phase="planning"
                    )

                    yield StreamEvent.create(
                        StreamEventType.AGENT_FINISH,
                        self.name,
                        result=error_msg
                    )

                    self.add_message(Message(input_text, "user"))
                    self.add_message(Message(error_msg, "assistant"))
                    return

                yield StreamEvent.create(
                    StreamEventType.STEP_FINISH,
                    self.name,
                    phase="planning",
                    plan=plan,
                    total_steps=len(plan)
                )

                statuses = [STEP_PENDING] * len(plan)
                results = [""] * len(plan)
                interrupted_idx = 0
                resumed_session_id = ""

            # ---- 执行阶段 ----
            total = len(plan)

            for i in range(interrupted_idx, total):
                step_num = i + 1
                step_description = plan[i]

                # 步骤开始（恢复时标注 resumed）
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    phase="execution",
                    step=step_num,
                    total_steps=total,
                    description=step_description,
                    resumed=bool(recovery_prompt) and i == interrupted_idx
                )

                print(f"\n--- 步骤 {step_num}/{total} ---")
                print(f"📋 {step_description}")

                # 标记进行中并写检查点
                statuses[i] = STEP_IN_PROGRESS
                self._checkpoint(saved_question, plan, statuses, results, current_step_index=i, interrupted=False)

                # 构建执行提示（恢复时注入断点恢复提示段）
                context = "\n".join([
                    f"步骤 {j+1}: {plan[j]} -> {results[j]}"
                    for j in range(i)
                    if statuses[j] == STEP_COMPLETED
                ])

                prompt = f"""原始问题: {saved_question}

完整计划:
{chr(10).join([f"{j+1}. {s}" for j, s in enumerate(plan)])}

已完成的步骤:
{context if context else "无"}

当前步骤: {step_description}

请执行当前步骤并给出结果。"""

                if recovery_prompt:
                    prompt = f"{recovery_prompt}\n\n{prompt}"

                messages = [{"role": "user", "content": prompt}]

                # 流式执行步骤
                step_result = ""
                async for chunk in self.llm.astream_invoke(messages, **kwargs):
                    step_result += chunk

                    yield StreamEvent.create(
                        StreamEventType.LLM_CHUNK,
                        self.name,
                        chunk=chunk,
                        phase="execution",
                        step=step_num
                    )

                    print(chunk, end="", flush=True)

                print()  # 换行

                # 标记完成并写检查点
                statuses[i] = STEP_COMPLETED
                results[i] = step_result
                self._checkpoint(saved_question, plan, statuses, results, current_step_index=i, interrupted=False)

                # 步骤完成
                yield StreamEvent.create(
                    StreamEventType.STEP_FINISH,
                    self.name,
                    phase="execution",
                    step=step_num,
                    result=step_result
                )

            # 生成最终答案
            yield StreamEvent.create(
                StreamEventType.STEP_START,
                self.name,
                phase="final_answer",
                description="生成最终答案"
            )

            final_prompt = f"""原始问题: {saved_question}

执行计划和结果:
{chr(10).join([f"{i+1}. {plan[i]} -> {results[i]}" for i in range(len(plan))])}

请基于以上步骤的执行结果，给出原始问题的最终答案。"""

            final_messages = [{"role": "user", "content": final_prompt}]

            final_answer = ""
            async for chunk in self.llm.astream_invoke(final_messages, **kwargs):
                final_answer += chunk

                yield StreamEvent.create(
                    StreamEventType.LLM_CHUNK,
                    self.name,
                    chunk=chunk,
                    phase="final_answer"
                )

            # 发送完成事件
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                result=final_answer,
                total_steps=total
            )

            print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

            # 全部完成，清理检查点（包括恢复来源文件，可能属于其他会话）
            self._clear_checkpoint()
            if resumed_session_id and resumed_session_id != self._session_id:
                try:
                    source_store = PlanCheckpointStore(
                        session_id=resumed_session_id,
                        checkpoint_dir=self.config.plan_checkpoint_dir
                    )
                    source_store.clear()
                except Exception:
                    pass

            # 保存到历史
            self.add_message(Message(saved_question, "user"))
            self.add_message(Message(final_answer, "assistant"))

        except KeyboardInterrupt:
            # 用户中断时保存检查点
            print("\n⚠️ 用户中断，自动保存计划检查点...")
            self._save_checkpoint_silent(saved_question, plan, statuses, results, interrupted=True)
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error="用户中断",
                error_type="KeyboardInterrupt"
            )
            raise

        except Exception as e:
            # 错误时保存检查点
            print(f"\n❌ 发生错误: {e}，自动保存计划检查点...")
            self._save_checkpoint_silent(saved_question, plan, statuses, results, interrupted=True)

            # 发送错误事件
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
