package au.gov.nla.dnn.training;

public interface TrainingListener
{
    public void onEpochComplete(int epoch, int bestEpoch, double bestEpochScore);
}