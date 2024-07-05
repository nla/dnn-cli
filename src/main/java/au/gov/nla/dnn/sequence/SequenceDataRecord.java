package au.gov.nla.dnn.sequence;

import java.io.Serializable;

public class SequenceDataRecord implements Serializable
{
    private static final long serialVersionUID = 1L;
    
    private double[] features;
    private double[] labels;
    
    public SequenceDataRecord(double[] features, double[] labels)
    {
        this.features = features;
        this.labels = labels;
    }

    public double[] getFeatures()
    {
        return features;
    }

    public double[] getLabels()
    {
        return labels;
    }
}