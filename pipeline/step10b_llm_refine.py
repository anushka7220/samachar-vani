import ollama


class LocalSummaryRefiner:

    def __init__(self, model="llama3:8b"):
        self.model = model

    def refine(self, headline: str, summary: str) -> str:

        if not summary.strip():
            return summary

        prompt = f"""
        आप एक पेशेवर हिंदी समाचार संपादक हैं।

        कार्य:
        नीचे दिए गए समाचार सारांश को ठीक करें।

        सख्त नियम:
        - केवल शुद्ध हिंदी में उत्तर दें।
        - अंग्रेज़ी का प्रयोग बिल्कुल न करें।
        - कोई व्याख्या, नोट, निर्देश या अतिरिक्त पाठ न लिखें।
        - "शीर्षक:" या "सारांश:" जैसे शब्द न लिखें।
        - केवल सुधरा हुआ समाचार सारांश लिखें।
        - 3 से 4 वाक्य ही लिखें।
        - शीर्षक को दोहराएँ नहीं।

        शीर्षक:
        {headline}

        सारांश:
        {summary}

        सुधरा हुआ सारांश:
        """

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response["message"]["content"].strip()

            return text

        except Exception as e:
            print("[LLM] refinement failed:", e)
            return summary