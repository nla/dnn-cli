package au.gov.nla.dnn.model;

import java.io.Serializable;

import au.gov.nla.dnn.sequence.InputSequenceInstance;

public class SerializedModelState implements Serializable
{
    private static final long serialVersionUID = 1L;
    
    private byte[] model;
    private InputSequenceInstance sequenceInstance;
    private String builderClass;
    private String[] labels;
    
    public SerializedModelState(byte[] model, InputSequenceInstance sequenceInstance, String builderClass, String[] labels)
    {
        this.model = model;
        this.sequenceInstance = sequenceInstance;
        this.builderClass = builderClass;
        this.labels = labels;
    }

    public byte[] getModel()
    {
        return model;
    }

    public InputSequenceInstance getSequenceInstance()
    {
        return sequenceInstance;
    }

    public String getBuilderClass()
    {
        return builderClass;
    }

    public String[] getLabels()
    {
        return labels;
    }
}