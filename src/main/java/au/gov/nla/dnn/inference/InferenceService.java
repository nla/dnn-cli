package au.gov.nla.dnn.inference;

public interface InferenceService
{
    public InferenceResult infer(String modelId, byte[] record) throws Exception;
}