package au.gov.nla.dnn.evaluation;

import org.deeplearning4j.earlystopping.scorecalc.ClassificationScoreCalculator;
import org.nd4j.evaluation.classification.Evaluation.Metric;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import au.gov.nla.dnn.training.TrainingHyperParameters;

public class MetricEvaluationScoreCalculator extends ClassificationScoreCalculator
{
    private static final long serialVersionUID = 1L;
    
    private TrainingHyperParameters hyperParameters;
    private int labelCount;
    
    public MetricEvaluationScoreCalculator(TrainingHyperParameters hyperParameters, int labelCount, DataSetIterator iterator)
    {
        super(getEvaluationMetric(hyperParameters), iterator);
        this.hyperParameters = hyperParameters;
        this.labelCount = labelCount;
    }
    
    protected double finalScore(Evaluation e)
    {
        return calculateScore(e, hyperParameters, labelCount);
    }

    public static double calculateScore(Evaluation e, TrainingHyperParameters hyperParameters, int labelCount)
    {
        double totalFp = 0;
        double totalFn = 0;
        double totalTp = 0;
        
        for(int i=0; i<labelCount; i++)
        {
            int fp = e.falsePositives().get(i);
            int fn = e.falseNegatives().get(i);
            int tp = e.truePositives().get(i);
            
            totalFp = totalFp+fp;
            totalFn = totalFn+fn;
            totalTp = totalTp+tp;
        }
        
        double precision = totalTp/(totalTp+totalFp);
        double recall = totalTp/(totalTp+totalFn);
        double f1 = (2d*precision*recall)/(precision+recall);
        
        switch(hyperParameters.getEvaluationMetric())
        {
            case TrainingHyperParameters.EVALUATION_METRIC_PRECISION:
            {
                return precision;
            }
            case TrainingHyperParameters.EVALUATION_METRIC_RECALL:
            {
                return recall;
            }
            case TrainingHyperParameters.EVALUATION_METRIC_F1:
            {
                return f1;
            }
            default:
            {
                return f1;
            }
        }
    }

    public static Metric getEvaluationMetric(TrainingHyperParameters hyperParameters)
    {
        switch(hyperParameters.getEvaluationMetric())
        {
            case TrainingHyperParameters.EVALUATION_METRIC_F1:
            {
                return Metric.F1;
            }
            case TrainingHyperParameters.EVALUATION_METRIC_RECALL:
            {
                return Metric.RECALL;
            }
            case TrainingHyperParameters.EVALUATION_METRIC_PRECISION:
            {
                return Metric.PRECISION;
            }
            default:
            {
                return Metric.F1;
            }
        }
    }
}