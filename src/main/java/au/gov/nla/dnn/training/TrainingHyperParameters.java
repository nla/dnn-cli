package au.gov.nla.dnn.training;

public class TrainingHyperParameters
{
    public static final int EVALUATION_METRIC_F1 = 0;
    public static final int EVALUATION_METRIC_PRECISION = 1;
    public static final int EVALUATION_METRIC_RECALL = 2;
    
    private int maxEpochs;
    private int batchSize;
    private long randomSeed;
    private int evaluationMetric;
    
    public TrainingHyperParameters(int maxEpochs, int batchSize, long randomSeed, int evaluationMetric)
    {
        this.maxEpochs = maxEpochs;
        this.batchSize = batchSize;
        this.randomSeed = randomSeed;
        this.evaluationMetric = evaluationMetric;
    }

    public int getMaxEpochs()
    {
        return maxEpochs;
    }

    public int getBatchSize()
    {
        return batchSize;
    }

    public long getRandomSeed()
    {
        return randomSeed;
    }

    public int getEvaluationMetric()
    {
        return evaluationMetric;
    }
}