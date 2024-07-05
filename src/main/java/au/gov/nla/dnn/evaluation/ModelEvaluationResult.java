package au.gov.nla.dnn.evaluation;

import java.util.LinkedHashMap;

import org.json.JSONObject;

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
    
    public JSONObject toJSON()
    {
        JSONObject a = new JSONObject();
        
        for(String label: accuracyRatings.keySet())
        {
            a.put(label, accuracyRatings.get(label).doubleValue());
        }
        
        JSONObject c = new JSONObject();
        
        for(String label: confusionMatrix.keySet())
        {
            ConfusionStatistics stats = confusionMatrix.get(label);
            JSONObject s = new JSONObject();
            
            s.put("true-positives", stats.getTp());
            s.put("true-negatives", stats.getTn());
            s.put("false-positives", stats.getFp());
            s.put("false-negatives", stats.getFn());
            
            s.put("precision", stats.getPrecision());
            s.put("recall", stats.getRecall());
            s.put("false-discovery-rate", stats.getFalseDiscoveryRate());
            s.put("false-positive-rate", stats.getFalsePositiveRate());
            s.put("matthews-correlation-coefficient", stats.getMcc());
            
            c.put(label, s);
        }
        
        JSONObject o = new JSONObject();
        o.put("accuracy", accuracy);
        o.put("accuracy-per-label", a);
        o.put("confusion-per-label", c);
        
        return o;
    }
}