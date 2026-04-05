#!/usr/bin/env python3
"""
CrewAI CLI Workflow 示例

使用 CLI 后端（claude/codex/gemini）运行 CrewAI 工作流，
无需 API key，直接使用本地已登录的 CLI 工具。

用法:
    python example_cli_workflow.py --topic "AI Agent 框架" --backend claude
    python example_cli_workflow.py --topic "量化投资" --backend gemini
"""

import argparse
import sys
import os

# Add the fork to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crewai import Agent, Task, Crew, Process


def create_research_crew(topic: str, backend: str = "claude") -> Crew:
    """创建一个研究型工作流 Crew。

    流水线: 研究员 → 评审员 → 写手

    Args:
        topic: 研究主题
        backend: CLI 后端 (claude/codex/gemini)
    """
    llm_model = f"cli/{backend}"

    # ========== Agent 定义 ==========

    researcher = Agent(
        role="行业研究员",
        goal=f"深入研究 {topic} 的最新趋势和关键发现",
        backstory=(
            "你是一位资深的行业分析师，擅长从多个维度分析行业趋势，"
            "善于发现别人忽视的关键洞察。你的分析总是有数据支撑、逻辑清晰。"
        ),
        llm=llm_model,
        verbose=True,
        allow_delegation=False,
    )

    critic = Agent(
        role="批判性评审员",
        goal="审查研究报告的质量，找出逻辑漏洞和遗漏",
        backstory=(
            "你是一位严格的学术评审员，以苛刻著称。"
            "你擅长找出论证中的逻辑漏洞、数据偏差、和结论过度推导。"
            "你的评审意见总是具体、可执行的。"
        ),
        llm=llm_model,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="内容写手",
        goal="基于研究和评审意见，撰写高质量的分析报告",
        backstory=(
            "你是一位专业的内容创作者，擅长将复杂的研究结论转化为"
            "清晰、有说服力的文章。你的写作风格简洁有力，"
            "善于用具体案例和数据说明问题。"
        ),
        llm=llm_model,
        verbose=True,
        allow_delegation=False,
    )

    # ========== Task 定义（Sequential 流水线）==========

    research_task = Task(
        description=(
            f"请深入研究「{topic}」这个领域：\n"
            f"1. 当前市场规模和增长趋势\n"
            f"2. 主要玩家和竞争格局\n"
            f"3. 关键技术突破和创新方向\n"
            f"4. 潜在风险和挑战\n"
            f"5. 未来 2-3 年的发展预测\n"
            f"\n请提供具体的数据和案例支撑你的分析。"
        ),
        expected_output="一份结构清晰的研究报告，包含上述 5 个维度的分析",
        agent=researcher,
    )

    review_task = Task(
        description=(
            "请严格审查上述研究报告：\n"
            "1. 论证逻辑是否自洽？\n"
            "2. 数据是否可靠，有无偏差？\n"
            "3. 结论是否有过度推导？\n"
            "4. 是否有重要维度被遗漏？\n"
            "5. 给出具体的修改建议\n"
            "\n请直接指出问题并给出改进方向。"
        ),
        expected_output="一份评审意见，包含具体问题和改进建议",
        agent=critic,
    )

    write_task = Task(
        description=(
            f"基于研究报告和评审意见，撰写一篇关于「{topic}」的深度分析文章：\n"
            f"1. 吸收评审意见中合理的修改建议\n"
            f"2. 保持研究报告中有价值的洞察\n"
            f"3. 用清晰的结构呈现（标题、小标题、要点）\n"
            f"4. 适合发布在专业媒体上的质量标准\n"
            f"\n输出一篇 800-1200 字的中文分析文章。"
        ),
        expected_output="一篇结构完整、论证充分的深度分析文章",
        agent=writer,
        output_file=f"output_{topic.replace(' ', '_')}.md",
    )

    # ========== Crew 组装 ==========

    crew = Crew(
        agents=[researcher, critic, writer],
        tasks=[research_task, review_task, write_task],
        process=Process.sequential,  # 流水线模式
        verbose=True,
    )

    return crew


def main():
    parser = argparse.ArgumentParser(
        description="CrewAI CLI Workflow - 使用 CLI 后端运行多 Agent 工作流"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="AI Agent 框架的发展趋势",
        help="研究主题",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="claude",
        choices=["claude", "codex", "gemini"],
        help="CLI 后端选择",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CrewAI CLI Workflow")
    print(f"  主题: {args.topic}")
    print(f"  后端: cli/{args.backend}")
    print(f"{'='*60}\n")

    crew = create_research_crew(topic=args.topic, backend=args.backend)
    result = crew.kickoff()

    print(f"\n{'='*60}")
    print(f"  最终输出")
    print(f"{'='*60}\n")
    print(result.raw)


if __name__ == "__main__":
    main()
