import { JsonView, allExpanded, darkStyles, defaultStyles } from 'react-json-view-lite';
import 'react-json-view-lite/dist/index.css';

export default function JSONView({jsoner}){
    return(
        <div className='bg-black'>
            <JsonView data={jsoner} shouldExpandNode={allExpanded} style={darkStyles} />
        </div>
    )
}