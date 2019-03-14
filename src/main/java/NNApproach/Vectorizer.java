package NNApproach;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.util.Collection;

/**
 * Create wordvectors based on own events
 *
 * @author Thomas Buchegger
 */
public class Vectorizer
{
    /**
     * Load existing wordvectors
     *
     * @param vectorPath
     * @return
     */
    public static Word2Vec LoadVectors(String vectorPath)
    {
        File vecs = new File(vectorPath);

        Word2Vec vec = WordVectorSerializer.readWord2VecModel(vecs);

        return vec;
    }

    /**
     * Create wordvectors
     *
     * @param vecInput      Text based on which the vectors should be created
     * @param vecOutputPath Path to where vector-file should be saved
     * @return
     */
    public static void CreateVectors(String vecInput, String vecOutputPath)
    {
        SentenceIterator iter = new LineSentenceIterator(new File(vecInput));
        iter.setPreProcessor(new SentencePreProcessor()
        {
            @Override
            public String preProcess(String s)
            {
                return s.toLowerCase();
            }
        });

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(5)
            .iterations(1)
            .layerSize(100)
            .seed(42)
            .windowSize(5)
            .iterate(iter)
            .tokenizerFactory(tokenizerFactory)
            .build();

        vec.fit();

        WordVectorSerializer.writeWord2VecModel(vec, vecOutputPath);

        Collection<String> lst = vec.wordsNearest("sport", 10);
        System.out.println(lst);
    }
}
