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
        transition={{ delay: 0.5 }}
      >
        I'm your virtual accounting guide.
      </motion.div>

      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="text-xl text-zinc-500 md:text-2xl"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.6 }}
      >
        Ask me anything about our accounting services.
      </motion.div>

      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="mt-6 text-sm text-zinc-400 md:text-base"
        exit={{ opacity: 0, y: 10 }}
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.8 }}
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
        <span className="font-medium text-zinc-900">Next.js</span>,{" "}
        <span className="font-medium text-zinc-900">TypeScript</span>,{" "}
        <span className="font-medium text-zinc-900">Python</span>,{" "}
        <span className="font-medium text-zinc-900">FastAPI</span>, and{" "}
        <span className="font-medium text-zinc-900">LangGraph</span>.
      </motion.div>

    </div>
  );
};

