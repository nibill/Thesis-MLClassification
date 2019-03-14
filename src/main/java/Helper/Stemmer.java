package Helper;

import opennlp.tools.stemmer.snowball.SnowballStemmer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Stem text with the SnowballStemmer
 *
 * @author Thomas Buchegger
 */
public class Stemmer
{
    /**
     * Stem a stream of string
     *
     * @param text Sting to stem
     * @return Stemmed stream
     */
    public static Stream<String> Stem(String text)
    {
        SnowballStemmer snowballStemmer = new SnowballStemmer(SnowballStemmer.ALGORITHM.GERMAN);

        Stream<String> stemmedText = (Stream<String>) snowballStemmer.stem((CharSequence) text);

        return stemmedText;
    }

    /**
     * Stem a single string
     *
     * @param word Word to stem
     * @return Stemmed word
     */
    public static String StemWord(String word)
    {
        SnowballStemmer snowballStemmer = new SnowballStemmer(SnowballStemmer.ALGORITHM.GERMAN);

        String stemmedWord = (String) snowballStemmer.stem(word);

        return stemmedWord;
    }

    /**
     * Stem a list of string
     *
     * @param wordList List with words to stem
     * @return Stemmed list
     */
    public static List<String> StemList(List<String> wordList)
    {
        List<String> stemmedList = new ArrayList<>();
        for (String word : wordList)
        {
            word = StemWord(word);
            stemmedList.add(word);
        }
        return stemmedList;
    }
}
