package Helper;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Remove stopwords from events
 *
 * @author Thomas Buchegger
 */
public class Stopwords
{
    /**
     * Remove stopwords from events
     *
     * @param events List with events
     * @return List with events without stopwords
     */
    public static ArrayList<String> RemoveStopwords(List<String> events)
    {

        List<String> stopwordList;
        Path filePath = Paths.get("Resources/wordsToDelete.txt");
        Charset charset = Charset.forName("ISO-8859-1");
        try
        {
            // load stopwords
            stopwordList = Files.readAllLines(filePath, charset);
        } catch (IOException e)
        {
            throw new RuntimeException("Cannot read stopwords file");
        }

        // convert stopwords list into set for faster lookup
        HashSet<String> stopwords = new HashSet<>(stopwordList);
        ArrayList<String> wordsWithoutStopwords = new ArrayList<>();

        // iterate through eventlist
        for (String event : events)
        {
            // split event into single words
            String[] words = event.split(" ");
            ArrayList<String> eventsWithoutStopwords = new ArrayList<>();

            // iterate through words
            for (String word : words)
            {
                // only add words which are not in the stopwords list
                if (!stopwords.contains(word))
                {
                    eventsWithoutStopwords.add(word);
                }
            }
            StringBuilder sb = new StringBuilder();
            for (String s : eventsWithoutStopwords)
            {
                sb.append(s);
                sb.append(" ");
            }

            // add all remaining words to a single event
            wordsWithoutStopwords.add(sb.toString());
        }

        return wordsWithoutStopwords;
    }
}
