from peak import Agent, CyclicBehaviour, JoinGroup, Message, Template
from src.mas.settings import group
import logging
import src.mas.model as model
from jsonpickle import decode, encode
logger = logging.getLogger(__name__)

class client(Agent):
    class InitFederation(CyclicBehaviour):
        async def on_start(self) -> None:
            await self.execute(JoinGroup(group, f"conference.{self.agent.jid.domain}"))
            self.toolbox = model.init()
            
        async def run(self) -> None:
            msg = await self.receive(60)
            if msg:
                logger.info(msg.get_metadata("population"))
                population = decode(msg.get_metadata("population"))
                logger.info(str(population[0]))
                round = msg.get_metadata("round")
                hof = model.train(self.toolbox, population)
                hof_message = Message(to=msg.sender)
                hof_message.body = f"Ended evolution {round}"
                hof_message.set_metadata("hof", encode(hof))
                hof_message.set_metadata("resource", "fl")
            else:
                self.kill()

        async def on_end(self) -> None:
            await self.agent.stop()


    async def setup(self) -> None:
        temp = Template()
        temp.set_metadata("resource", "fl")
        self.add_behaviour(self.InitFederation(), temp)
