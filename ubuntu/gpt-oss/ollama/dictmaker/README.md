🎯 Project Overview
This project aims to create comprehensive, age-categorized English dictionaries using AI language models. Our goal is to contribute to educational diversity by generating alternative English dictionary resources that complement existing traditional dictionaries.
The project leverages local AI models through Ollama to collect, categorize, and enhance English vocabulary with Korean translations, age-appropriate classifications, and usage examples - creating valuable resources for language learners and educators.
🌟 Key Features

Massive Scale Collection: Capable of generating dictionaries with 15,000+ words
Age-Based Classification: Words categorized by learning levels (Elementary, Middle School, High School, Adult, Professional)
Multilingual Support: English words with Korean translations and examples
Automated Processing: Two-stage pipeline for efficient large-scale dictionary generation
Prefix-Based Collection: Advanced collection strategy using alphabetical prefixes (aa, ab, ac...) for comprehensive coverage
Structured Output: JSON format for easy integration and further processing

🏗️ Architecture
Two-Stage Pipeline
Stage 1: Basic Word Collection

Collects raw English words using prefix-based approach
Supports multiple collection modes (adaptive, 2-letter, 3-letter)
Handles up to 676 prefixes (aa-zz) for maximum coverage
Automatic deduplication and validation

Stage 2: Enhanced Information

Adds Korean translations, age classifications, and usage examples
Processes words in chunks for efficiency
Generates comprehensive statistics and metadata

🚀 Quick Start
Prerequisites
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull your preferred model (example)
ollama pull gpt-oss:120b
Installation
git clone https://github.com/EXCorpDEV/repo_hajunho.slack.com.git 
cd ai-english-dictionary
pip install requests
Basic Usage
# Stage 1: Collect basic words (adaptive mode - recommended for beginners)
python3 massive_dict.py --stage 1 --mode adaptive --batch 30

# Stage 2: Enhance with detailed information
python3 massive_dict.py --stage 2 --file [generated_file]
Advanced Usage
# Large-scale dictionary generation (15,000+ words)
python3 massive_dict.py --stage 1 --mode 2letter --batch 35

# Custom model and settings
python3 massive_dict.py --stage 1 --model your-model:latest --batch 50

# Automated full pipeline
./auto_massive_dict.sh
📊 Collection Modes
ModePrefixesExpected WordsTimeUse Caseadaptive~601,500-3,0001-2 hoursBalanced, high-quality collection2letter67615,000-25,00020-30 hoursComprehensive dictionary3letter~2,60010,000-15,00015-25 hoursSpecialized coverage
📁 Output Structure
json{
  "metadata": {
    "title": "Enhanced English-Korean Dictionary",
    "total_words": 15420,
    "statistics": {
      "by_age_group": {
        "elementary": 3200,
        "middle": 2800,
        "high": 4100,
        "adult": 3920,
        "professional": 1400
      }
    }
  },
  "words": [
    {
      "word": "serendipity",
      "korean": "뜻밖의 발견",
      "age_group": "adult",
      "part_of_speech": "noun",
      "example": "Finding this book was pure serendipity.",
      "difficulty": 4
    }
  ]
}
🎓 Educational Impact
This project addresses the need for diverse educational resources by:

Democratizing Dictionary Creation: Making comprehensive dictionary generation accessible
Supporting Multiple Learning Levels: Age-appropriate word classification
Enhancing Language Learning: Bilingual examples and contextual usage
Promoting Open Education: Freely available, customizable dictionary resources

🛠️ Technical Details
Performance Optimization

Chunk Processing: Processes multiple words simultaneously to reduce API calls
Automatic Resume: Can resume interrupted operations
Rate Limiting: Built-in delays to prevent API overload
Error Handling: Robust error recovery and reporting

Customization Options

Model Selection: Compatible with various Ollama models
Batch Sizes: Adjustable for different performance requirements
Language Pairs: Extensible to other language combinations
Age Categories: Customizable classification systems

🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.
Areas for Contribution

Additional language pairs
Improved age classification algorithms
Performance optimizations
Educational integration tools
Quality assessment metrics

🙏 Acknowledgments
This project is built upon the excellent work of several open-source communities:
Core Technologies

Ollama Project: For providing an excellent local AI model serving platform that makes large language models accessible to everyone. Special thanks to the Ollama team for their commitment to open-source AI infrastructure.
GPT-OSS Project: Gratitude to the open-source language model developers and the broader GPT ecosystem that enables projects like this to exist.

Development Stack

Python Community: For the robust ecosystem that powers this project
JSON Standard: For providing a universal data interchange format
Open Source Movement: For fostering collaboration and knowledge sharing

Special Recognition
We extend our heartfelt gratitude to:

The Ollama engineering team for creating tools that democratize AI access
Open-source LLM researchers who make advanced language understanding accessible
Educational technology pioneers who inspire projects that enhance learning
Contributors and users who help improve and expand this project

📈 Project Statistics

Supported Languages: English-Korean (extensible)
Maximum Dictionary Size: 25,000+ words
Age Categories: 5 levels (Elementary to Professional)
Processing Speed: ~2-5 seconds per word
Output Format: Structured JSON
Platform: Cross-platform (Linux, macOS, Windows)

🔮 Future Roadmap

 Support for additional language pairs
 Web-based interface for non-technical users
 Integration with popular learning management systems
 Advanced semantic analysis and word relationships
 Community-driven quality validation
 Mobile app for dictionary access
 API service for third-party integrations

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
📞 Contact

Issues: Please use GitHub Issues for bug reports and feature requests
Discussions: Join our GitHub Discussions for general questions
Email: [hajunho@excorp.io]


Made with ❤️ for the global education community
This project demonstrates the power of combining open-source AI tools with educational innovation to create resources that benefit learners worldwide.
