"""AI-асистент аналітика: розрахунки робить Pandas, LLM лише інтерпретує та формулює."""
from .eda import profile_dataframe, auto_eda
from .hypotheses import collect_facts, generate_hypotheses
from .nl2pandas import ask
from .explain_drop import explain_metric_drop
from .summary import executive_summary

__all__ = ['profile_dataframe', 'auto_eda', 'collect_facts', 'generate_hypotheses',
           'ask', 'explain_metric_drop', 'executive_summary']
