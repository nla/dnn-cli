package au.gov.nla.dnn.inference;

import java.util.HashMap;

import org.json.JSONArray;
import org.json.JSONObject;

public class InferenceResult
{
    private String highestLabel;
    private double highestConfidence;
    private HashMap<String, Double> scores;
    
    public InferenceResult(String highestLabel, double highestConfidence, HashMap<String, Double> scores)
    {
        this.highestLabel = highestLabel;
        this.highestConfidence = highestConfidence;
        this.scores = scores;
    }

    public String getHighestLabel()
    {
        return highestLabel;
    }

    public double getHighestConfidence()
    {
        return highestConfidence;
    }
    
    public HashMap<String, Double> getScores()
    {
        return scores;
    }

    public JSONObject toJSON()
    {
        JSONArray s = new JSONArray();
        
        for(String label: getScores().keySet())
        {
            JSONObject o = new JSONObject();
            o.put(label, getScores().get(label));
            s.put(o);
        }
        
        JSONObject j = new JSONObject();
        j.put("highest-label", getHighestLabel());
        j.put("highest-confidence", getHighestConfidence());
        j.put("scores", s);
        
        return j;
    }
}