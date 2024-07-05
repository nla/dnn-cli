package au.gov.nla.dnn.sequence;

import java.util.List;

import org.json.JSONObject;

public interface InputSequence
{
    public InputSequenceInstance generateInstance(List<String> labels, JSONObject config) throws Exception;
}