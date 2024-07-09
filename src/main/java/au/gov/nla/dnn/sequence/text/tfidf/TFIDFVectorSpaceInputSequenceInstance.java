package au.gov.nla.dnn.sequence.text.tfidf;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Properties;

import au.gov.nla.dnn.record.RawDataRecord;
import au.gov.nla.dnn.record.RawDataRecordProvider;
import au.gov.nla.dnn.sequence.InputSequenceInstance;
import au.gov.nla.dnn.sequence.SequenceDataRecord;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

public class TFIDFVectorSpaceInputSequenceInstance implements InputSequenceInstance
{
    private static final long serialVersionUID = 1L;
    
    private String wordSplitPattern;
    private List<String> exclusionPatterns;
    private int minCharacters;
    private int maxCharacters;
    private int minOccurrances;
    private List<String> labelList;
    
    private LinkedHashMap<String, Word> words;
    private double totalProcessedRecords;
    
    public TFIDFVectorSpaceInputSequenceInstance(String wordSplitPattern, List<String> exclusionPatterns, 
            int minCharacters, int maxCharacters, int minOccurrances, List<String> labelList)
    {
        this.wordSplitPattern = wordSplitPattern;
        this.exclusionPatterns = exclusionPatterns;
        this.minCharacters = minCharacters;
        this.maxCharacters = maxCharacters;
        this.minOccurrances = minOccurrances;
        this.labelList = labelList;
        this.words = new LinkedHashMap<String, Word>();
    }

    public int getFeatureCount()
    {
        return words.size();
    }
    
    public int getLabelCount()
    {
        return labelList.size();
    }
    
    public void preProcess(long randomSeed, RawDataRecordProvider recordProvider) throws Exception
    {
        boolean exclude;
        words.clear();
        
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
        
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        
        while(recordProvider.hasMoreRecords())
        {
            RawDataRecord record = recordProvider.getNextRecord();
            totalProcessedRecords++;
            
            // Build list of unique words
            
            HashSet<String> wordSet = new HashSet<String>();
            String data = new String(record.getData());
            
            for(String segment: data.split(wordSplitPattern))
            {
                if(segment.length()>=minCharacters && segment.length()<=maxCharacters)
                {
                    wordSet.add(segment.toLowerCase());
                }
            }
            
            // Lemmatize unique words
            
            HashSet<String> lemmatizedSet = new HashSet<String>();
            StringBuilder b = new StringBuilder();
            
            for(String w: wordSet)
            {
                b.append(w).append(" ");
            }
            
            Annotation annotation = new Annotation(b.toString());
            pipeline.annotate(annotation);
            
            for(CoreLabel token: annotation.get(CoreAnnotations.TokensAnnotation.class))
            {
                String lemma = token.lemma();
                lemmatizedSet.add(lemma==null?token.originalText():lemma);
            }
            
            // Exclude words and add remaining to vocab
            
            for(String w: lemmatizedSet)
            {
                exclude = false;
                
                for(String pattern: exclusionPatterns)
                {
                    if(w.matches(pattern))
                    {
                        exclude = true;
                        break;
                    }
                }
                if(!exclude)
                {
                    Word word = getWord(w);
                    word.setRecordsContainingWord(word.getRecordsContainingWord()+1);
                }
            }
            
            System.out.println("[TFIDFVectorSpaceInputSequenceInstance] Preprocessed text record: "+totalProcessedRecords);
        }
        
        for(String w: new ArrayList<String>(words.keySet()))
        {
            if(words.get(w).getRecordsContainingWord()<minOccurrances)
            {
                words.remove(w);
            }
        }
        
        System.out.println("[TFIDFVectorSpaceInputSequenceInstance] Total words: "+words.size());
    }
    
    public SequenceDataRecord process(RawDataRecord record)
    {
        HashMap<String, Integer> wordFrequency = new HashMap<String, Integer>();
        String featureText = new String(record.getData());
        
        for(String segment: featureText.split("\\s+"))
        {
            if(segment.length()>=minCharacters)
            {
                if(wordFrequency.containsKey(segment))
                {
                    wordFrequency.put(segment, wordFrequency.get(segment)+1);
                }
                else
                {
                    wordFrequency.put(segment, 1);
                }
            }
        }
        
        double wordCount = wordFrequency.size();
        double[] features = new double[words.size()];
        int wordIndex = 0;
        
        for(String wordKey: words.keySet())
        {
            Word word = words.get(wordKey);
            Integer frequency = wordFrequency.get(wordKey);
            
            if(frequency!=null && frequency>0)
            {
                features[wordIndex] = (((double)frequency)/wordCount)*(Math.log(totalProcessedRecords/((double)word.getRecordsContainingWord())));
            }
            else
            {
                features[wordIndex] = 0d;
            }
            
            wordIndex++;
        }
        
        System.out.println("[TFIDFVectorSpaceInputSequenceInstance] Processed text record: "+features);
        return new SequenceDataRecord(features, record.getLabels());
    }
    
    private Word getWord(String value)
    {
        Word word = words.get(value);
        
        if(word==null)
        {
            word = new Word(value);
            words.put(value, word);
        }
        
        return word;
    }
    
    protected static class Word implements Serializable
    {
        private static final long serialVersionUID = 1L;
        private String value;
        private int recordsContainingWord;
        
        protected Word(String value)
        {
            this.value = value;
            this.recordsContainingWord = 0;
        }

        public int getRecordsContainingWord()
        {
            return recordsContainingWord;
        }

        public void setRecordsContainingWord(int recordsContainingWord)
        {
            this.recordsContainingWord = recordsContainingWord;
        }

        public String getValue()
        {
            return value;
        }
    }
}
