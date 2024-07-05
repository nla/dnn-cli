package au.gov.nla.dnn.sequence;

import java.io.Serializable;

import au.gov.nla.dnn.record.RawDataRecord;
import au.gov.nla.dnn.record.RawDataRecordProvider;

public interface InputSequenceInstance extends Serializable
{
    public int getFeatureCount();
    public int getLabelCount();
    
    public void preProcess(long randomSeed, RawDataRecordProvider recordProvider) throws Exception;
    public SequenceDataRecord process(RawDataRecord record) throws Exception;
}
