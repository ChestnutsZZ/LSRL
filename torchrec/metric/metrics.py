from torchrec.metric.Hit import Hit
from torchrec.metric.IMetric import IMetric
from torchrec.metric.NDCG import NDCG


def get_metric(metric_name: str) -> IMetric:
    """根据评价指标获取实例"""
    if not isinstance(metric_name, str):
        raise ValueError(f"metric_name参数不合法: {metric_name}")
    class_name, k = metric_name.split('@')
    k = int(k)
    if class_name == "ndcg":
        return NDCG(99, k)
    elif class_name == "hit":
        return Hit(99, k)
    else:
        raise ValueError(f"metric_name参数不合法: {metric_name}")
