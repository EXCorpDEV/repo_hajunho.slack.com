import React, {useState, useEffect} from 'react';

function App() {
  const [count, setCount] = useState(1);
  const [name, setName] = useState('')

  const handleCountUpdate = () => {
    setCount(count + 1);
  };

  // 마운트 + [ item ] 변경될때만 실행
  useEffect(() => {
    console.log('count 변화');
  }, [count]);

  const handleInputChange = (e) => {
    setName(e.target.value);
  };



return (
  <div>
    <button onClick={handleCountUpdate}>Update</button>
    <span>count: {count}</span>
    <input type="text" value={name} onChange={handleInputChange} />
    <span>name : {name}</span>
  </div>
  );
}

export default App;