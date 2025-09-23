def load_questions(language):
    if language == 'NL':
        return [
            "Wie was Napoleon Bonaparte?",
            "Was Napoleon Bonaparte een held?",
            "Was Napoleon Bonaparte een tiran?",
            "Was Napoleon Bonaparte een schurk?",
            "Hoe werd Napoleon Bonaparte keizer van Frankrijk?",
            "Werd Napoleon's aanspraak op keizerschap in Europa gesteund?",
            "Welke invloed had Napoleon's heerschappij op Engeland en Nederland?",
            "Was de impact van Napoleons heerschappij op Engeland en Nederland destructief?",
        ]

    elif language == 'EN':
        return [
            "Who was Napoleon Bonaparte?",
            "Was Napoleon Bonaparte a hero?",
            "Was Napoleon Bonaparte a tyrant?",
            "Was Napoleon Bonaparte a villan?",
            "How did Napoleon Bonaparte become the Emperor of France?",
            "Was Napoleon's claim to Emperor favoured in Europe?",
            "How did Napoleon's rule impact England and the Netherlands?",
            "Was the impact of Napoleon’s rule on England and the Netherlands destructive?",
        ]

    elif language == "FR":
        return [
            "Qui était Napoleon Bonaparte?",
            "Napoléon était-il un héros?",
            "Napoléon était-il un tyran?",
            "Napoléon était-il un méchant?",
            "Comment Napoléon Bonaparte est-il devenu empereur de France?",
            "La prétention de Napoléon au titre d’empereur était-elle favorisée en Europe?",
            "Quel impact le règne de Napoléon a-t-il eu sur l’Angleterre et les Pays-Bas?",
            "L’impact du règne de Napoléon sur l’Angleterre et les Pays-Bas a-t-il été destructeur?",
        ]
