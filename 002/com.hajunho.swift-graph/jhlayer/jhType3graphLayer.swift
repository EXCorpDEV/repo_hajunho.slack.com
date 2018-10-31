//
//  jhType3graphLayer.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 31/10/2018.
//  Copyright © 2018 hajunho.com. All rights reserved.
//

import UIKit

class jhType3graphLayer<T> : jhCommonDataLayer<T>, jhLayer_p {
    
    //TODO: escaping hardcoding later on
    private let mMargin : CGFloat = 45
    private let hhhWidth : CGFloat = 86400 //24h
    private let maxY : CGFloat = 15 //10000:300=15:0.45
    
    override func draw(in ctx: CGContext) {
        "".pwd(self)
        let isTestMode: Bool = true
        let xRatio : CGFloat = self.bounds.width / hhhWidth
        
        guard
            var ctime = (self.superScene as? jhSceneTimeLine)?.currentTime,
            var etime = (self.superScene as? jhSceneTimeLine)?.endTime
            else { return }
        
        if GS.shared.logLevel.contains(.network2) {
            print("ctime in jhType3graphLayer<T>\(ctime)")
            print("ctime oh no actually etime in jhType3graphLayer<T>", etime)
        }
        
        panelID = 2 //TODO: management should be in the same class which network class related with it
        mValuesOfDatas.removeAll()
        
        guard let jhDatas = jhDataCenter.mDatas[panelID] else {
            return
        }
        
        //        var maxY : CGFloat = 0
        //        for man in 0..<jhDatas.d.count {
        //            if jhDatas.d[man].y > maxY { maxY = jhDatas.d[man].y }
        //        }
        
        let yRatio = self.bounds.height / maxY
        
        for man in 0..<jhDatas.d.count {
            let x = (jhDatas.d[man] as! ssss).x
            let y = (jhDatas.d[man] as! ssss).y
            
            let x2 = (jhDatas.d[man] as! ssss).x2
            let y2 = (jhDatas.d[man] as! ssss).y2
            
            var fx = ((CGFloat(x) - CGFloat(etime.timeIntervalSince1970)) * xRatio)
            let fy = CGFloat(y) * yRatio
            
            var fx2 = ((CGFloat(x2) - CGFloat(etime.timeIntervalSince1970)) * xRatio)
            let fy2 = CGFloat(y2) * yRatio
            
            if(GS.shared.logLevel.contains(.json)) {
                print("json values is changed to (fx,fy) (", fx, ")    (", fy, ") in Panel")
            }
            
            fx = fx * (1410/1500) + mMargin //TODO: panel size 1500(*4?)
            fx2 = fx2 * (1410/1500) + mMargin //TODO: panel size 1500(*4?)
            
            if x >= CGFloat(ctime.timeIntervalSince1970) || x <= CGFloat(etime.timeIntervalSince1970) {
                continue
            } else {
                //////////////////DEBUG//////////////////////////////////////////////////////////////////////////////
                //                drawPoint(ctx, fx, fy, 2, 2, thickness: 3, UIColor.blue.cgColor)                 //
                //                drawPoint(ctx, fx2, fy2, 2, 2, thickness: 3, UIColor.red.cgColor)                //
                //                print("draw fx, fy", getX(fx)!, " ", getY(fy)!)                                  //
                /////////////////////////////////////////////////////////////////////////////////////////////////////
                //                drawRectCustom(context: ctx, x1: fx, y1: fy, x2: fx2, y2: fy2)
                
                ctx.setFillColor(UIColor(red: 196, green: 134, blue: 237).cgColor)
                ctx.setStrokeColor(UIColor(red: 213, green: 180, blue: 234).cgColor)
                ctx.setLineWidth(1)
                
                let rectangle = CGRect(x: fx, y: 0.45*yRatio, width: fx2-fx, height: fy)
                ctx.addRect(rectangle)
                ctx.drawPath(using: .fillStroke)
            }
        }
        //            jhDataCenter.mDatas[panelID]!
        if isTestMode {
            testModeDrawing(in: ctx)
        }
    }
    
    func getX(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * self.bounds.width / jhDraw.ARQ
        return retX
    }
    
    func getY(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * self.bounds.width / jhDraw.ARQ
        return retY
    }
    
    func drawLine(_ context : CGContext, _ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat) {
        jhDraw.worldLine(context: context, x1, y1, x2, y2, 1, UIColor.red.cgColor)
    }
    
    func drawRectCustom(context: CGContext, x1: CGFloat, y1: CGFloat, x2: CGFloat, y2: CGFloat) {
        drawLine(context, x1, y1, x2, y1)
        drawLine(context, x2, y1, x2, 0)
        drawLine(context, x2, 0, x1, 0)
        drawLine(context, x1, 0, x1, y1)
    }
    
    //    func drawRect(_ context: CGContext, margin : CGFloat, color : CGColor) {
    ////        mColor = color
    //        drawRect(context, margin: margin)
    //    }
    
    
    func testModeDrawing(in ctx: CGContext) {
        var x : Int = 0
        var pointCloud = Array<CGPoint>()
        var fx, fy : CGFloat
        
        for man in jhDataCenter.nonNetworkData {
            mValuesOfDatas.append(man)
        }
        
        for y in mValuesOfDatas { //TODO:
            //ref:drawLine(CGFloat(x)*axisDistance + mMargin, mMargin, CGFloat(x) * axisDistance + mMargin, 10000-mMargin)
            x += 1
            fx = CGFloat(x) * xDistance
            fy = CGFloat(y) * self.bounds.height / jhDraw.ARQ + GV.s.ui_common_margin
            drawTestPoint(ctx, fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
            pointCloud.append(CGPoint.init(x: getXonVPanel(fx+GV.s.ui_common_margin)!, y: getYonVPanel(fy)!))
        }
        
        ctx.move(to: CGPoint.init(x: 0, y: 0))
        ctx.setStrokeColorSpace(CGColorSpaceCreateDeviceRGB())
        ctx.setStrokeColor(UIColor.blue.cgColor)
        ctx.setLineWidth(1.0)
        ctx.addLines(between: pointCloud)
        ctx.strokePath()
    }
    
    override func drawPoint(_ ctx: CGContext, _ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, thickness : CGFloat, _ color : CGColor){
        jhDraw.worldEllipse(context: ctx, x, y, width, height, thickness, color)
        //        x++GV.s.ui_common_margin
    }
    
    func drawTestPoint(_ ctx: CGContext, _ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, thickness : CGFloat, _ color : CGColor){
        //        worldEllipse(context: mContext, getX(x)!, getY(jhDraw.maxR - y)!, width, height, thickness, color)
        if GS.shared.logLevel.contains(.graph) {
            print("worldEllipse(context: mContext,", getXonVPanel(x+GV.s.ui_common_margin)!, getYonVPanel(jhDraw.ARQ-y)!, width, height, thickness, color)
        }
        jhDraw.worldEllipse(context: ctx, getXonVPanel(x+GV.s.ui_common_margin)!, getYonVPanel(y)!, width, height, thickness, color)
    }
    
    func getXonVPanel(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * self.bounds.width / jhDraw.ARQ
        return retX
    }
    
    func getYonVPanel(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * self.bounds.width / jhDraw.ARQ
        return retY
    }
}


