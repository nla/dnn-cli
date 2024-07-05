package au.gov.nla.dnn.record;

public class RawDataRecord
{
    private byte[] data;
    private double[] labels;
    
    public RawDataRecord(byte[] data, double[] labels)
    {
        this.data = data;
        this.labels = labels;
    }

    public byte[] getData()
    {
        return data;
    }

    public double[] getLabels()
    {
        return labels;
    }
}
