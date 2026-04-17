import React, { useState } from 'react';
import { HelpCircle, Target, Brain, TrendingUp, BarChart3, Trophy, Lightbulb, MessageCircle, ChevronDown, ChevronUp } from 'lucide-react';

const FAQItem = ({ question, answer, darkMode, icon: Icon }) => {
  const [isOpen, setIsOpen] = useState(false);
  const textPrimary = darkMode ? 'text-white' : 'text-slate-900';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';
  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';

  return (
    <div className={`${cardBg} border rounded-lg sm:rounded-xl overflow-hidden hover:shadow-lg transition-all`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-4 sm:p-6 flex items-center justify-between text-left gap-3"
      >
        <div className="flex items-center space-x-3 sm:space-x-4 flex-1 min-w-0">
          <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center flex-shrink-0">
            <Icon size={16} className="text-white sm:w-5 sm:h-5" />
          </div>
          <h3 className={`text-sm sm:text-base md:text-lg font-semibold ${textPrimary} break-words`}>{question}</h3>
        </div>
        <div className="flex-shrink-0">
          {isOpen ? <ChevronUp size={20} className={`${textSecondary} sm:w-6 sm:h-6`} /> : <ChevronDown size={20} className={`${textSecondary} sm:w-6 sm:h-6`} />}
        </div>
      </button>
      {isOpen && (
        <div className={`px-4 sm:px-6 pb-4 sm:pb-6 ${textSecondary} leading-relaxed text-sm sm:text-base`}>
          {answer}
        </div>
      )}
    </div>
  );
};

export default function Help({ darkMode }) {
  const cardBg = darkMode ? 'bg-slate-800/90 backdrop-blur-lg border-slate-700' : 'bg-white/90 backdrop-blur-lg border-blue-100';
  const textPrimary = darkMode ? 'text-white' : 'text-slate-900';
  const textSecondary = darkMode ? 'text-slate-300' : 'text-slate-600';

  const faqs = [
    {
      icon: Target,
      question: "How do I make a prediction?",
      answer: (
        <div className="space-y-2">
          <p>Making predictions is easy:</p>
          <ol className="list-decimal list-inside space-y-1 ml-2">
            <li>Navigate to the <strong className="text-blue-400">Predictor</strong> tab</li>
            <li>Choose between <strong className="text-blue-400">Predictions</strong> (upcoming matches) or <strong className="text-blue-400">Results & Accuracy</strong> (completed matches)</li>
            <li>Select the gameweek using the navigation buttons</li>
            <li>Click <strong className="text-blue-400">Predict Match</strong> on any fixture</li>
            <li>View detailed predictions including predicted score and win probabilities</li>
          </ol>
        </div>
      )
    },
    {
      icon: TrendingUp,
      question: "What do the colors mean in Predictor?",
      answer: (
        <div className="space-y-2">
          <p>We use color coding throughout the app:</p>
          <ul className="space-y-1 ml-2">
            <li><span className="text-green-400 font-bold">● Green</span> - Home team win predictions</li>
            <li><span className="text-yellow-400 font-bold">● Yellow</span> - Draw predictions</li>
            <li><span className="text-teal-400 font-bold">● Teal</span> - Away team win predictions</li>
            <li><span className="text-red-500 font-bold">● Red</span> - Wrong predictions (in Results & Accuracy)</li>
          </ul>
        </div>
      )
    },
    {
      icon: BarChart3,
      question: "How accurate are the predictions in Bhakundo?",
      answer: (
        <div className="space-y-2">https://bhakundo-backend.onrender.com/
          <p>Our model achieves:</p>
          <ul className="space-y-1 ml-2">
            <li>• <strong className="text-blue-400">High score-outcome alignment</strong> (predicted scores match actual results)</li>
            <li>• <strong className="text-blue-400">High outcome accuracy</strong> (predicting win/draw/loss)</li>
            <li>• Trained on <strong className="text-blue-400"> Premier League matches From 2010 to 2026</strong></li>
          </ul>
        </div>
      )
    },
    {
      icon: Trophy,
      question: "Can I edit team formations?",
      answer: (
        <div className="space-y-2">
          <p>Yes! In the Predictions tab:</p>
          <ol className="list-decimal list-inside space-y-1 ml-2">
            <li>Click the <strong className="text-blue-400">Edit Formation</strong> button on any match</li>
            <li>Select formations for both home and away teams</li>
            <li>Click <strong className="text-blue-400">Predict Match</strong> to see how tactics affect the outcome</li>
          </ol>
          <p className="pt-2">Formation changes affect the tactical analysis in predictions.</p>
        </div>
      )
    },
    {
      icon: Brain,
      question: "What makes predictions reliable and nearly accurate?",
      answer: (
        <div className="space-y-3">
          <p>Bhakundo uses a <strong className="text-blue-400">multi-stage ensemble system</strong> with 3 specialized models:</p>
          <div className="space-y-2 ml-2">
            <div>
              <strong className="text-green-400"> Base Model</strong>
              <p className="ml-4">Analyzes team form, league position, venue stats, rest days, and head-to-head records</p>
            </div>
            <div>
              <strong className="text-yellow-400"> Lineup Model</strong>
              <p className="ml-4">Adds tactical analysis including formations, midfield battles, and aggression scores</p>
            </div>
            <div>
              <strong className="text-teal-400">Score Model</strong>
              <p className="ml-4">Predicts exact match scores with 83.4% alignment to actual outcomes</p>
            </div>
          </div>
          <p className="pt-2">The system combines all three predictions with confidence weighting to give you the most accurate forecast!</p>
        </div>
      )
    },
    {
      icon: HelpCircle,
      question: "What's the difference between Predictions and Results?",
      answer: (
        <div className="space-y-2">
          <p><strong className="text-blue-400">Predictions Tab:</strong></p>
          <ul className="list-disc list-inside ml-2 mb-3">
            <li>Shows upcoming fixtures for the selected gameweek</li>
            <li>Make predictions before matches are played</li>
            <li>Edit formations to test tactical scenarios</li>
          </ul>
          <p><strong className="text-green-400">Results & Accuracy Tab:</strong></p>
          <ul className="list-disc list-inside ml-2">
            <li>Shows completed matches with actual scores</li>
            <li>Compare predictions against real results</li>
            <li>See gameweek accuracy percentage automatically</li>
          </ul>
        </div>
      )
    },
    {
      icon: Lightbulb,
      question: "Where can I see prediction accuracy?",
      answer: (
        <div className="space-y-2">
          <p>Accuracy is shown in the <strong className="text-blue-400">Results & Accuracy</strong> tab:</p>
          <ul className="space-y-2 ml-2">
            <li>
              <strong className="text-blue-400"> GW Accuracy Card</strong>
              <p className="ml-4 text-sm">At the top, showing overall gameweek prediction accuracy percentage</p>
            </li>
            <li>
              <strong className="text-green-400"> Individual Match Cards</strong>
              <p className="ml-4 text-sm">Click "Show Prediction" to see if each prediction was correct</p>
            </li>
            <li>
              <strong className="text-yellow-400">Color Coding</strong>
              <p className="ml-4 text-sm">Green border = Correct, Red border = Wrong</p>
            </li>
          </ul>
        </div>
      )
    },
    {
      icon: Trophy,
      question: "What does 'Score Matched' vs 'Score Didn't Match' mean?",
      answer: (
        <div className="space-y-2">
          <p>When viewing predictions in Results & Accuracy:</p>
          <ul className="space-y-2 ml-2">
            <li>
              <strong className="text-green-400">✓ Prediction was Correct</strong>
              <p className="ml-4 text-sm">We predicted the right outcome (which team won or if it was a draw)</p>
              <p className="ml-4 text-sm mt-1">
                • <strong className="text-teal-400"> Score matched Exactly</strong> - We got both the outcome AND exact score right!<br/>
                • <strong>Score didn't match</strong> - We got the outcome right but the exact score was different
              </p>
            </li>
            <li className="pt-2">
              <strong className="text-red-500">✗ Prediction was Wrong</strong>
              <p className="ml-4 text-sm">We predicted the wrong outcome (e.g., predicted home win but away won)</p>
            </li>
          </ul>
        </div>
      )
    },
    {
      icon: Brain,
      question: "Why does the predicted score sometimes show a different result than the outcome prediction?",
      answer: (
        <div className="space-y-3">
          <p>Great question! This happens because we use <strong className="text-blue-400">three separate AI models</strong> that work together:</p>
          
          <div className="space-y-3 ml-2">
            <div className="border-l-4 border-blue-500 pl-4">
              <strong className="text-blue-400">1. Base Model</strong>
              <p className="text-sm mt-1">Analyzes team statistics, form, head-to-head records, and league position. Predicts: <strong>Home Win / Draw / Away Win</strong></p>
            </div>
            
            <div className="border-l-4 border-green-500 pl-4">
              <strong className="text-green-400">2. Lineup Model</strong>
              <p className="text-sm mt-1">Considers team formations, player availability, and tactical setup. Also predicts: <strong>Home Win / Draw / Away Win</strong></p>
            </div>
            
            <div className="border-l-4 border-teal-500 pl-4">
              <strong className="text-teal-400">3. Score Model</strong>
              <p className="text-sm mt-1">Predicts the actual scoreline (e.g., 2-1, 0-0). This is trained separately to estimate goals scored.</p>
            </div>
          </div>

          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mt-3">
            <p className="text-yellow-400 font-semibold mb-2">🔄 How They Work Together:</p>
            <p className="text-sm">
              The <strong>final "Bhakundo Predicts"</strong> outcome is determined by combining all three models using our ensemble algorithm. 
              If Base and Lineup models strongly agree on "Draw" (e.g., 43% and 62%), that becomes the final prediction - even if the 
              Score model initially predicted 1-5. We then <strong>adjust the score to match</strong> the ensemble outcome for consistency.
            </p>
          </div>

          <p className="text-sm italic pt-2">
            This multi-model approach gives <strong>higher accuracy</strong> than any single model alone!
          </p>
        </div>
      )
    },
    {
      icon: MessageCircle,
      question: "How do I contact support?",
      answer: (
        <div className="space-y-2">
          <p>Need help or have feedback?</p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li>Visit the <strong className="text-blue-400">Home</strong> page</li>
            <li>Scroll to the <strong className="text-blue-400">Contact</strong> section</li>
            <li>Fill out the form with your message</li>
            <li>We'll respond as soon as possible!</li>
          </ul>
        </div>
      )
    }
  ];

  return (
    <div className="px-6 md:px-12 py-12">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center shadow-xl">
              <HelpCircle size={40} className="text-white" />
            </div>
          </div>
          <h1 className={`text-5xl font-bold ${textPrimary} mb-4`}>Help & Guide</h1>
          <p className={`text-xl ${textSecondary}`}>Everything you need to know about Bhakundo Predictor</p>
        </div>

        {/* FAQ Section */}
        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <FAQItem
              key={index}
              question={faq.question}
              answer={faq.answer}
              icon={faq.icon}
              darkMode={darkMode}
            />
          ))}
        </div>

        {/* Footer Note */}
        <div className={`${cardBg} border rounded-xl p-8 mt-8 text-center`}>
          <p className={`${textSecondary} text-lg`}>
            Still have questions? Visit the <strong className="text-blue-400">Home</strong> page and send us a message!
          </p>
        </div>
      </div>
    </div>
  );
}
