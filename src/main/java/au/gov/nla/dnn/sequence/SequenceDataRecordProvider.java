package au.gov.nla.dnn.sequence;

import java.util.List;

public interface SequenceDataRecordProvider
{
    public void reset() throws Exception;
    public boolean hasMoreRecords();
    public List<SequenceDataRecord> getMoreRecords(int count) throws Exception;
    public int getTotalRecords();
    public void dispose();
}