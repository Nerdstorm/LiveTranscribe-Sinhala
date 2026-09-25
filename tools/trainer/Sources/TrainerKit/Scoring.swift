import Foundation

/// Error rates as scripts/make_replay.py computes them, so the trainer's numbers and the replay
/// set's are the same measure: text normalised (NFKC, case folded, one apostrophe, no separators
/// inside numbers), then words, or characters that are letters, marks or digits (for Sinhala,
/// the vowel signs and the virama count; spaces, punctuation and zero-width joiners don't).
public enum Scoring {
    static let apostrophes: [Unicode.Scalar: Unicode.Scalar] = ["\u{2019}": "'", "\u{2018}": "'", "\u{02BC}": "'"]
    // "1,000", "1.000" and "1 000" are one number, and "3,5" matches "3.5".
    nonisolated(unsafe) static let numberSeparator = try! NSRegularExpression(
        pattern: #"(?<=\d)(?:[.,]|\s(?=\d{3}(?!\d)))(?=\d)"#
    )

    public static func normalise(_ text: String) -> String {
        let folded = text.precomposedStringWithCompatibilityMapping.folding(options: .caseInsensitive, locale: nil)
        var unified = String.UnicodeScalarView()
        unified.append(contentsOf: folded.unicodeScalars.map { apostrophes[$0] ?? $0 })
        let result = String(unified)
        return numberSeparator.stringByReplacingMatches(
            in: result, range: NSRange(result.startIndex..., in: result), withTemplate: ""
        )
    }

    /// Whether the error rates count a character: letters, marks and digits do.
    public static func counts(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
             .nonspacingMark, .spacingMark, .enclosingMark,
             .decimalNumber, .letterNumber, .otherNumber:
            true
        default:
            false
        }
    }

    /// The words of a transcript. An apostrophe inside a word stays part of it; other punctuation
    /// separates words.
    public static func words(_ text: String) -> [String] {
        var spaced = String.UnicodeScalarView()
        for scalar in normalise(text).unicodeScalars {
            spaced.append(counts(scalar) || scalar == "'" ? scalar : " ")
        }
        return String(spaced).split(separator: " ")
            .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "'")) }
            .filter { !$0.isEmpty }
    }

    /// The characters of a transcript (Unicode scalars, as Python counts them), without spaces or
    /// punctuation.
    public static func characters(_ text: String) -> [Unicode.Scalar] {
        normalise(text).unicodeScalars.filter(counts)
    }

    /// The fewest substitutions, deletions and insertions that turn `reference` into `hypothesis`.
    public static func editDistance<T: Equatable>(_ reference: [T], _ hypothesis: [T]) -> Int {
        var previous = Array(0 ... hypothesis.count)
        var current = previous
        for (i, expected) in reference.enumerated() {
            current[0] = i + 1
            for (j, actual) in hypothesis.enumerated() {
                current[j + 1] = min(previous[j + 1] + 1, current[j] + 1, previous[j] + (expected == actual ? 0 : 1))
            }
            swap(&previous, &current)
        }
        return previous[hypothesis.count]
    }

    /// Whether a word is written in Latin letters: an English word in a Sinhala transcript.
    public static func isLatin(_ word: String) -> Bool {
        var letters = 0
        for scalar in word.unicodeScalars where scalar.properties.isAlphabetic {
            letters += 1
            switch scalar.value {
            case 0x41 ... 0x5A, 0x61 ... 0x7A, 0xC0 ... 0x24F, 0x1E00 ... 0x1EFF: continue
            default: return false
            }
        }
        return letters > 0
    }
}

/// Error counts over a set of transcripts: errors are summed and divided by the references'
/// total length, so long references weigh more than short ones.
public struct ErrorTally: Codable, Equatable, Sendable {
    public var characterErrors = 0
    public var characters = 0
    public var wordErrors = 0
    public var words = 0
    public var utterances = 0
    /// Hypotheses with under half the reference's characters: a transcript cut short.
    public var truncated = 0
    /// Latin-letter words: in the references, in the hypotheses, and in both (counted as a multiset).
    public var latinInReferences = 0
    public var latinInHypotheses = 0
    public var latinMatched = 0

    public init() {}

    public mutating func add(reference: String, hypothesis: String) {
        let expected = Scoring.characters(reference), actual = Scoring.characters(hypothesis)
        characterErrors += Scoring.editDistance(expected, actual)
        characters += expected.count
        let expectedWords = Scoring.words(reference), actualWords = Scoring.words(hypothesis)
        wordErrors += Scoring.editDistance(expectedWords, actualWords)
        words += expectedWords.count
        utterances += 1
        if actual.count * 2 < expected.count { truncated += 1 }

        let referenceLatin = expectedWords.filter(Scoring.isLatin)
        var hypothesisLatin = actualWords.filter(Scoring.isLatin)
        latinInReferences += referenceLatin.count
        latinInHypotheses += hypothesisLatin.count
        for word in referenceLatin {
            if let index = hypothesisLatin.firstIndex(of: word) {
                hypothesisLatin.remove(at: index)
                latinMatched += 1
            }
        }
    }

    public var characterErrorRate: Double { characters > 0 ? Double(characterErrors) / Double(characters) : 0 }
    public var wordErrorRate: Double { words > 0 ? Double(wordErrors) / Double(words) : 0 }
    /// Of the references' English words, the share the hypotheses wrote in English letters.
    public var latinRecall: Double? { latinInReferences > 0 ? Double(latinMatched) / Double(latinInReferences) : nil }
    /// Of the hypotheses' English words, the share the references have.
    public var latinPrecision: Double? { latinInHypotheses > 0 ? Double(latinMatched) / Double(latinInHypotheses) : nil }
}
