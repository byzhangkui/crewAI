#!/usr/bin/env python3
"""
CrewAI CLI Flow 示例

使用 Flow（高级流水线）编排多个 Crew，
每个阶段可以使用不同的 CLI 后端。

用法:
    python example_cli_flow.py --topic "新能源汽车"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, listen, start


class ResearchState(BaseModel):
    """Flow 的共享状态"""
    topic: str = ""
    research_output: str = ""
    review_output: str = ""
    final_article: str = ""


class ResearchFlow(Flow[ResearchState]):
    """三阶段研究流水线: 研究 → 评审 → 撰写

    每个阶段是一个独立的 Crew，通过 Flow 的 @start/@listen 串联。
    """

    @start()
    def research_phase(self):
        """第一阶段: 研究"""
        print(f"\n[Phase 1] 研究阶段 - 主题: {self.state.topic}")

        researcher = Agent(
            role="研究员",
            goal=f"深入研究 {self.state.topic}",
            backstory="你是资深行业分析师，善于发现关键洞察。",
            llm="cli/claude",
            verbose=True,
        )

        task = Task(
            description=f"研究「{self.state.topic}」的现状、趋势和关键发现。提供数据支撑。",
            expected_output="一份简洁的研究摘要（300-500字）",
            agent=researcher,
        )

        crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.research_output = result.raw
        return result.raw

    @listen(research_phase)
    def review_phase(self, research_output):
        """第二阶段: 评审"""
        print(f"\n[Phase 2] 评审阶段")

        critic = Agent(
            role="评审员",
            goal="找出研究报告中的逻辑漏洞和改进空间",
            backstory="你是严格的学术评审员，以苛刻著称。",
            llm="cli/claude",
            verbose=True,
        )

        task = Task(
            description=(
                f"评审以下研究报告，指出问题和改进建议：\n\n"
                f"---\n{self.state.research_output}\n---"
            ),
            expected_output="评审意见，包含具体问题和修改建议",
            agent=critic,
        )

        crew = Crew(agents=[critic], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.review_output = result.raw
        return result.raw

    @listen(review_phase)
    def write_phase(self, review_output):
        """第三阶段: 撰写"""
        print(f"\n[Phase 3] 撰写阶段")

        writer = Agent(
            role="写手",
            goal="撰写高质量的分析文章",
            backstory="你是专业内容创作者，擅长将复杂研究转化为清晰文章。",
            llm="cli/claude",
            verbose=True,
        )

        task = Task(
            description=(
                f"基于以下研究和评审，撰写一篇关于「{self.state.topic}」的深度文章：\n\n"
                f"[研究报告]\n{self.state.research_output}\n\n"
                f"[评审意见]\n{self.state.review_output}\n\n"
                f"要求：800-1200字，结构清晰，适合专业媒体发布。"
            ),
            expected_output="一篇完整的深度分析文章",
            agent=writer,
        )

        crew = Crew(agents=[writer], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.final_article = result.raw
        return result.raw


def main():
    parser = argparse.ArgumentParser(description="CrewAI CLI Flow Pipeline")
    parser.add_argument("--topic", type=str, default="AI Agent 框架的发展趋势")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CrewAI CLI Flow Pipeline")
    print(f"  主题: {args.topic}")
    print(f"{'='*60}")

    flow = ResearchFlow()
    flow.state.topic = args.topic
    result = flow.kickoff()

    print(f"\n{'='*60}")
    print(f"  最终输出")
    print(f"{'='*60}\n")
    print(result)


if __name__ == "__main__":
    main()
