package au.gov.nla.dnn.evaluation;

import java.util.LinkedHashMap;

public class ModelEvaluationResult
{
    private double accuracy;
    private LinkedHashMap<String, Double> accuracyRatings;
    private LinkedHashMap<String, ConfusionStatistics> confusionMatrix;
    
    public ModelEvaluationResult(double accuracy, LinkedHashMap<String, Double> accuracyRatings, LinkedHashMap<String, ConfusionStatistics> confusionMatrix)
    {
        this.accuracy = accuracy;
        this.accuracyRatings = accuracyRatings;
        this.confusionMatrix = confusionMatrix;
    }
    
    public double getAccuracy()
    {
        return accuracy;
    }
    
    public LinkedHashMap<String, Double> getAccuracyRatings()
    {
        return accuracyRatings;
    }
    
    public LinkedHashMap<String, ConfusionStatistics> getConfusionMatrix()
    {
        return confusionMatrix;
    }
    
    public static class ConfusionStatistics
    {
        private int tp;
        private int tn;
        private int fp;
        private int fn;
        private double precision;
        private double recall;
        private double falseDiscoveryRate;
        private double falsePositiveRate;
        private double mcc;
        
        public ConfusionStatistics(int tp, int tn, int fp, int fn, double precision, double recall, double falseDiscoveryRate, double falsePositiveRate, double mcc)
        {
            this.tp = tp;
            this.tn = tn;
            this.fp = fp;
            this.fn = fn;
            this.precision = precision;
            this.recall = recall;
            this.falseDiscoveryRate = falseDiscoveryRate;
            this.falsePositiveRate = falsePositiveRate;
            this.mcc = mcc;
        }

        public int getTp()
        {
            return tp;
        }

        public int getTn()
        {
            return tn;
        }

        public int getFp()
        {
            return fp;
        }

        public int getFn()
        {
            return fn;
        }

        public double getPrecision()
        {
            return precision;
        }

        public double getRecall()
        {
            return recall;
        }

        public double getFalseDiscoveryRate()
        {
            return falseDiscoveryRate;
        }

        public double getFalsePositiveRate()
        {
            return falsePositiveRate;
        }

        public double getMcc()
        {
            return mcc;
        }

        public void setMcc(double mcc)
        {
            this.mcc = mcc;
        }
    }
}