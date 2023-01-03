from peak import Agent, OneShotBehaviour, JoinGroup, Message, Template
from settings import group, n_clients, n_rounds
import asyncio
import logging
import model
from jsonpickle import decode, encode
logger = logging.getLogger("agent_server")

class server(Agent):
    class InitFederation(OneShotBehaviour):
        async def on_start(self) -> None:
            await self.execute(JoinGroup(group, f"conference.{self.agent.jid.domain}"))
            while not len(await self.agent.group_members(f"{group}@conference.{self.agent.jid.domain}")) >= n_clients:
                await asyncio.sleep(1)
            self.round = 0
            
        async def run(self) -> None:
            #init first population
            self.toolbox = model.init()
            population = self.toolbox.population(n=300)
            while self.round < n_rounds:
                self.round += 1
                msg = Message(to=f"{group}@conference.{self.agent.jid.domain}")
                msg.body = f"Init round {self.round}"
                msg.set_metadata("population", encode(population))
                msg.set_metadata("round", str(self.round))
                msg.set_metadata("resource", "fl")
                await self.send_to_group(msg)
                #aggregation
                population = model.tools.HallOfFame(10)
                while count := 0 < n_clients:
                    hof_message = await self.receive(60)
                    if hof_message:
                        population.update(decode(hof_message.get_metadata("hof")))
                    count += 1
            logger.info(f"Best Individual: {str(population[0])}")
            logger.info(f"Evaluation: {str(self.toolbox.evaluate(population[0]))}")

        async def on_end(self) -> None:
            await self.agent.stop()

    async def setup(self) -> None:
        temp = Template()
        temp.set_metadata("resource", "fl")
        self.add_behaviour(self.InitFederation(), temp)
