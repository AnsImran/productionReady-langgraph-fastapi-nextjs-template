import { motion } from "framer-motion";

export const Greeting = () => {
  return (
    <div
      className="mx-auto mt-4 flex size-full max-w-3xl flex-col justify-center px-4 md:mt-16 md:px-8"
      key="overview"
    >
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="font-semibold text-xl md:text-2xl"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.03 }}
      >
        I'm your virtual accounting guide.
      </motion.div>

      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="text-xl text-zinc-500 md:text-2xl"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.035 }}
      >
        Ask me anything about our accounting services.
      </motion.div>

      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="mt-6 text-sm text-zinc-400 md:text-base"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.04 }}
      >
        Created by{" "}
        <a
          href="https://github.com/AnsImran"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-blue-500 hover:text-blue-600 transition-colors"
        >
          Ans Imran
        </a>{" "}
        — built with{" "}
        <span className="font-medium text-zinc-900 dark:text-white">Next.js</span>,{" "}
        <span className="font-medium text-zinc-900 dark:text-white">TypeScript</span>,{" "}
        <span className="font-medium text-zinc-900 dark:text-white">Python</span>,{" "}
        <span className="font-medium text-zinc-900 dark:text-white">FastAPI</span>, and{" "}
        <span className="font-medium text-zinc-900 dark:text-white">LangChain</span>.
      </motion.div>


    </div>
  );
};

